#!/usr/bin/env python3
"""
TRUE production end-to-end pipeline, per raw image:
  RF-DETR detector -> car bbox (cascade) + antenna boxes
    -> crop car -> body UNet(coarse)+refiner ; holes UNet(coarse)+refiner
    -> per antenna box: antenna UNet + colfill
  -> assemble full-res masks: outline = body ∪ antenna ; holes ; antenna
Compares OURS vs PROD vs GT per class (boundary-F + IoU), caches masks for viz.

Crops come from the DETECTOR (not GT) — the real production number, so detection
error compounds into segmentation.

Run from repo root:
  PYTHONPATH=. /home/rutger/miniconda3/envs/cc_sam3/bin/python carcutter/car_bbox_detector/end2end_run.py --n 200
"""
import argparse, csv, os
from pathlib import Path
import numpy as np, cv2, torch
from PIL import Image
from scipy.ndimage import binary_fill_holes
import segmentation_models_pytorch as smp
import rfdetr
from carcutter.car_bbox_detector.seg_eval import boundary_f, mask_iou, MEAN, STD
from carcutter.car_bbox_detector.gen_coarse_masks import coarse_prob          # crops around a 0/255 bbox-mask -> full-res prob
from carcutter.car_bbox_detector.refine_infer_eval import refine             # tiled native-res boundary refine
from carcutter.car_bbox_detector.antenna_continuity import colfill           # per-column vertical fill
from carcutter.car_bbox_detector.cascade_eval import predict_union, TOP_PAD, SIDE_PAD
from carcutter.car_bbox_detector.birefnet_eval import load_birefnet, birefnet_outline
from carcutter.car_bbox_detector.antenna_fp_clean import keep_main

ROOT = "carcutter/car_bbox_detector"; EXP = f"{ROOT}/experiments"
CAR, ANT = 0, 1
BLUE, WHITE = (0, 0, 255), (255, 255, 255)
USE_HOLES_REFINER = True   # set from the post-train A/B (coarse-only vs coarse+refiner)
BIREFNET_CKPT = f"{ROOT}/birefnet/BiRefNet/ckpts/car_outline_v2/epoch_16.pth"  # v2: trunk-oversampled + pad0.15 (eliminates segment-drop tail)
TINT_CKPT = f"{ROOT}/birefnet/BiRefNet/ckpts/car_holes_v2/epoch_16.pth"        # window-hole ensemble partner
ANT_THR = 0.5   # antenna detection threshold (PR sweep: 0.5 -> precision 0.91 / recall 0.78; fewer stick FPs)
BODY_SIZE = 1024     # outline silhouette res (1024 = trained res; 1536 regresses BF@1 0.809->0.776)
PUNCHOUT_SIZE = 1536 # separate hi-res pass ONLY for punchout-hole extraction (small holes 40->54%)


def load_unet(ckpt, ch, enc, device):
    m = smp.Unet(enc, encoder_weights=None, in_channels=ch, classes=1).to(device).eval()
    st = torch.load(ckpt, map_location=device); m.load_state_dict(st.get("model", st)); return m


def boxmask(box, H, W):
    m = np.zeros((H, W), np.uint8); x0, y0, x1, y1 = [int(v) for v in box]
    m[max(0,y0):min(H,y1+1), max(0,x0):min(W,x1+1)] = 255; return m


def antenna_boxes(det, ax_thr=0.15):
    cls, conf, xyxy = np.asarray(det.class_id), np.asarray(det.confidence), np.asarray(det.xyxy)
    return [xyxy[i] for i in range(len(xyxy)) if cls[i] == ANT and conf[i] >= ax_thr]


def bbox_boxmask(mask, pad_frac, H, W):
    """0/255 box mask around `mask`'s bbox, expanded by pad_frac per side."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    ph, pw = int((y1-y0+1)*pad_frac), int((x1-x0+1)*pad_frac)
    y0, y1 = max(0, y0-ph), min(H-1, y1+ph); x0, x1 = max(0, x0-pw), min(W-1, x1+pw)
    m = np.zeros((H, W), np.uint8); m[y0:y1+1, x0:x1+1] = 255; return m


def holes_cascade(holes, holes_ref, img, carbox_mask, dev):
    """2-pass: rough holes on the car crop -> crop the window region -> re-run at
    the holes model's native framing -> refiner. Avoids the train/deploy crop
    mismatch (model was trained on tight window crops, not whole-car crops)."""
    H, W = img.shape[:2]
    p1 = coarse_prob(holes, img, carbox_mask, 768, dev) > 0.5      # rough, wide framing
    if p1.sum() < 30:
        return np.zeros((H, W), bool)
    box2 = bbox_boxmask(p1, 0.5, H, W)                            # window region (+50% context)
    p2prob = coarse_prob(holes, img, box2, 768, dev)              # native framing
    return refine(holes_ref, img, p2prob, dev) > 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=f"{ROOT}/data/index.csv")
    ap.add_argument("--out", default=f"{ROOT}/data/e2e")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--body", default="birefnet", choices=["birefnet", "unet"],
                    help="body silhouette model: birefnet (v2 hybrid) or unet (v1)")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); (out/"masks").mkdir(parents=True, exist_ok=True)

    det = rfdetr.RFDETRMedium.from_checkpoint(f"{EXP}/rfdetr_car_antenna_v1/checkpoint_best_total.pth")
    bfbody = load_birefnet(BIREFNET_CKPT, dev) if args.body == "birefnet" else None
    bftint = load_birefnet(TINT_CKPT, dev)   # window-hole ensemble partner (UNet ∪ BiRefNet-tint)
    body = load_unet(f"{EXP}/unet_car_seg_v1/checkpoints/best.pt", 3, "efficientnet-b4", dev)
    body_ref = load_unet(f"{EXP}/unet_refine_v1/checkpoints/best.pt", 4, "efficientnet-b2", dev)
    holes = load_unet(f"{EXP}/unet_holes_carcrop_v1/checkpoints/best.pt", 3, "efficientnet-b4", dev)
    holes_ref = load_unet(f"{EXP}/unet_refine_holes_v1/checkpoints/best.pt", 4, "efficientnet-b2", dev)
    ant = load_unet(f"{EXP}/unet_antenna_v1/checkpoints/best.pt", 3, "efficientnet-b4", dev)

    rows = [r for r in csv.DictReader(open(args.index)) if r["split"] == "test"][:args.n]
    fout = open(out/"e2e_metrics.csv", "w", newline=""); wr = csv.writer(fout)
    wr.writerow(["image", "cls", "method", "iou", "bf1", "bf2", "bf3"])

    def prod_path(r, kind):
        p = r["prod_outline"].replace("/outline/", f"/{kind}/") if r["prod_outline"] else ""
        return p if p and os.path.exists(p) else ""
    def prodmask(p):
        a = np.array(Image.open(p)); return a > 0 if a.ndim == 2 else a.sum(2) > 0
    def logrow(img, cls, method, gt, pr):
        wr.writerow([img, cls, method, f"{mask_iou(gt,pr):.4f}",
                     f"{boundary_f(gt,pr,1):.4f}", f"{boundary_f(gt,pr,2):.4f}", f"{boundary_f(gt,pr,3):.4f}"])

    for i, r in enumerate(rows):
        pil = Image.open(r["image"]).convert("RGB"); img = np.array(pil); H, W = img.shape[:2]
        rgb = np.array(Image.open(r["mask"]).convert("RGB"))
        gt = {"outline": rgb.sum(2) > 0, "holes": (rgb == BLUE).all(2), "antenna": (rgb == WHITE).all(2)}

        carbox = predict_union(det, pil, 0.3, 0.15)        # cascade car-extent box (crop region)
        if carbox is None:
            continue
        cm = boxmask(carbox, H, W)
        # body silhouette: BiRefNet (v2 hybrid winner) on the detector car crop, else UNet coarse->refine
        if bfbody is not None:
            body_ref_mask = birefnet_outline(bfbody, img, carbox, dev, size=BODY_SIZE)
        else:
            body_ref_mask = refine(body_ref, img, coarse_prob(body, img, cm, 768, dev), dev) > 0.5
        # holes: coarse on car crop (deploy framing, model trained to match) -> refine.
        # USE_HOLES_REFINER toggled after A/B; single-pass replaces the dead cascade.
        holes_coarse = coarse_prob(holes, img, cm, 768, dev)
        holes_unet = (refine(holes_ref, img, holes_coarse, dev) > 0.5) if USE_HOLES_REFINER \
                     else (holes_coarse > 0.5)
        # window-hole ENSEMBLE: UNet ∪ BiRefNet-tint (they miss different windows -> recall 73->81%)
        holes_mask = holes_unet | birefnet_outline(bftint, img, carbox, dev)
        # antenna: per detected antenna box -> antenna UNet -> colfill
        ant_mask = np.zeros((H, W), bool)
        for ab in antenna_boxes(det.predict(pil, threshold=ANT_THR), ax_thr=ANT_THR):
            ap_prob = coarse_prob(ant, img, boxmask(ab, H, W), 256, dev)
            ant_mask |= colfill(ap_prob > 0.5)
        # outline = body ∪ antenna, FP-cleaned (drop floating blobs not connected to the car)
        outline = keep_main(body_ref_mask | ant_mask, 9)

        ours = {"outline": outline, "holes": holes_mask, "antenna": ant_mask}
        prod = {"outline": prod_path(r, "outline"), "holes": prod_path(r, "holes")}
        for cls in ("outline", "holes", "antenna"):
            if gt[cls].sum() < 20:
                continue
            logrow(r["image"], cls, "ours", gt[cls], ours[cls])
            if cls in prod and prod[cls]:
                logrow(r["image"], cls, "prod", gt[cls], prodmask(prod[cls]))
        # punchout structural holes from a SEPARATE hi-res (1536) body pass — small holes survive
        # at higher res; kept separate so the 1024 outline boundary isn't regressed.
        if bfbody is not None:
            body_hires = birefnet_outline(bfbody, img, carbox, dev, size=PUNCHOUT_SIZE)
            outline_hires = keep_main(body_hires | ant_mask, 9)   # cleaned silhouette -> enclosed holes register
        else:
            outline_hires = outline
        punchout = binary_fill_holes(outline_hires) & ~outline_hires
        # cache for viz (uint8 packed)
        np.savez_compressed(out/"masks"/f"{Path(r['image']).stem}.npz",
                            outline=outline, holes=holes_mask, antenna=ant_mask,
                            punchout=punchout, carbox=np.array(carbox))
        if (i+1) % 25 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)
    fout.close()
    print(f"Done -> {out}/e2e_metrics.csv + masks/")


if __name__ == "__main__":
    main()
