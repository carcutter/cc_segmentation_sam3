#!/usr/bin/env python3
"""
UNIFIED production end-to-end pipeline — the single inference path from the Phase A/B/C work.
Per raw image:
  car-extent detector -> car box -> crop car
  BiRefNet -> body silhouette (outline) + tint windows (ensemble partner)
  ONE rfdetr_multiclass_v1 pass on the car crop -> antenna/hole/window region boxes
    class 0 antenna     -> antenna UNet (256) + colfill
    class 1 holeregion  -> hole-seg UNet  (512)  (2-stage small punchout holes)
    class 2 windowregion-> window-seg UNet(512)  (2-stage see-through windows)
  assemble:
    outline   = keep_main(body ∪ antenna)
    punchout  = fill_holes(body_hires ∪ antenna) & ~...   ∪  hole-2stage   (structural)
    windows   = BiRefNet-tint  ∪  window-2stage           (tint see-through)
    antenna   = antenna mask
Compares OURS vs PROD vs GT per class (boundary-F + IoU), caches masks for viz.

ONE region-detector pass replaces the three separate region detectors; the antenna UNet is
now driven by that same pass (class 0). Crops come from the DETECTOR (real production number).

Run from repo root:
  PYTHONPATH=. /home/rutger/miniconda3/envs/cc_sam3/bin/python carcutter/car_bbox_detector/end2end_unified.py --n 200
"""
import argparse, csv, os
from pathlib import Path
import numpy as np, cv2, torch
from PIL import Image
from scipy.ndimage import binary_fill_holes
import segmentation_models_pytorch as smp
import rfdetr
from carcutter.car_bbox_detector.seg_eval import boundary_f, mask_iou, MEAN, STD
from carcutter.car_bbox_detector.gen_coarse_masks import coarse_prob
from carcutter.car_bbox_detector.antenna_continuity import colfill
from carcutter.car_bbox_detector.cascade_eval import predict_union
from carcutter.car_bbox_detector.birefnet_eval import load_birefnet, birefnet_outline
from carcutter.car_bbox_detector.antenna_fp_clean import keep_main

ROOT = "carcutter/car_bbox_detector"; EXP = f"{ROOT}/experiments"
BLUE, WHITE = (0, 0, 255), (255, 255, 255)
BIREFNET_CKPT = f"{ROOT}/birefnet/BiRefNet/ckpts/car_outline_v2/epoch_16.pth"
TINT_CKPT = f"{ROOT}/birefnet/BiRefNet/ckpts/car_holes_v2/epoch_16.pth"
BODY_SIZE = 1024; PUNCHOUT_SIZE = 1536
# unified region detector + dedicated UNet operating points (from the 2-stage sweeps)
DET_THR = 0.5; SEG_HOLE = 256; SEG_REG = 512
HOLE_THR = 0.6; WIN_THR = 0.6; ANT_THR = 0.5
ANTENNA, HOLE, WINDOW = 0, 1, 2   # rfdetr_multiclass_v1 class_id is 0-indexed (=cat_id-1)


def load_unet(ckpt, enc, device):
    m = smp.Unet(enc, encoder_weights=None, in_channels=3, classes=1).to(device).eval()
    st = torch.load(ckpt, map_location=device); m.load_state_dict(st.get("model", st)); return m


def boxmask(box, H, W):
    m = np.zeros((H, W), np.uint8); x0, y0, x1, y1 = [int(v) for v in box]
    m[max(0,y0):min(H,y1+1), max(0,x0):min(W,x1+1)] = 255; return m


@torch.no_grad()
def segment_box(seg, img, box, dev, thr, sz):
    """Crop EXACTLY box -> sz -> sigmoid>thr -> resize back. Matches the 2-stage eval/train."""
    H, W = img.shape[:2]; x0, y0, x1, y1 = [int(v) for v in box]
    x0, y0 = max(0, x0), max(0, y0); x1, y1 = min(W, x1), min(H, y1)
    if x1 - x0 < 8 or y1 - y0 < 8: return None, None
    tile = cv2.resize(img[y0:y1, x0:x1], (sz, sz))
    t = torch.from_numpy(((tile / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]).float().to(dev)
    pr = (torch.sigmoid(seg(t))[0, 0] > thr).cpu().numpy().astype(np.uint8)
    pr = cv2.resize(pr, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST) > 0
    return pr, (x0, y0, x1, y1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=f"{ROOT}/data/index.csv")
    ap.add_argument("--out", default=f"{ROOT}/data/e2e_unified")
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); (out/"masks").mkdir(parents=True, exist_ok=True)

    cardet = rfdetr.RFDETRMedium.from_checkpoint(f"{EXP}/rfdetr_car_antenna_v1/checkpoint_best_total.pth")
    unidet = rfdetr.RFDETRMedium.from_checkpoint(f"{EXP}/rfdetr_multiclass_v1/checkpoint_best_ema.pth")
    bfbody = load_birefnet(BIREFNET_CKPT, dev)
    bftint = load_birefnet(TINT_CKPT, dev)
    ant = load_unet(f"{EXP}/unet_antenna_v1/checkpoints/best.pt", "efficientnet-b4", dev)
    holeseg = load_unet(f"{EXP}/unet_holeseg_v2/best.pt", "efficientnet-b2", dev)
    winseg = load_unet(f"{EXP}/unet_windowseg_v2/best.pt", "efficientnet-b2", dev)

    rows = [r for r in csv.DictReader(open(args.index)) if r["split"] == "test"][:args.n]
    fout = open(out/"e2e_metrics.csv", "w", newline=""); wr = csv.writer(fout)
    wr.writerow(["image", "cls", "method", "iou", "bf1", "bf2", "bf3"])

    def prod_path(r, kind):
        p = r["prod_outline"].replace("/outline/", f"/{kind}/") if r["prod_outline"] else ""
        return p if p and os.path.exists(p) else ""
    def prodmask(p):
        a = np.array(Image.open(p)); return a > 0 if a.ndim == 2 else a.sum(2) > 0
    def logrow(im, cls, method, gt, pr):
        wr.writerow([im, cls, method, f"{mask_iou(gt,pr):.4f}",
                     f"{boundary_f(gt,pr,1):.4f}", f"{boundary_f(gt,pr,2):.4f}", f"{boundary_f(gt,pr,3):.4f}"])

    for i, r in enumerate(rows):
        pil = Image.open(r["image"]).convert("RGB"); img = np.array(pil); H, W = img.shape[:2]
        rgb = np.array(Image.open(r["mask"]).convert("RGB"))
        gt = {"outline": rgb.sum(2) > 0, "holes": (rgb == BLUE).all(2), "antenna": (rgb == WHITE).all(2)}

        carbox = predict_union(cardet, pil, 0.3, 0.15)
        if carbox is None:
            continue
        cx0, cy0, cx1, cy1 = [int(v) for v in carbox]
        cx0, cy0 = max(0, cx0), max(0, cy0); cx1, cy1 = min(W, cx1), min(H, cy1)
        carcrop = img[cy0:cy1, cx0:cx1]
        if carcrop.shape[0] < 32 or carcrop.shape[1] < 32:
            continue

        # BiRefNet body silhouette + tint windows on the detector car crop
        body_mask = birefnet_outline(bfbody, img, carbox, dev, size=BODY_SIZE)
        tint_mask = birefnet_outline(bftint, img, carbox, dev)

        # === SINGLE unified region-detector pass -> 3 dedicated UNets ===
        dd = unidet.predict(Image.fromarray(carcrop), threshold=DET_THR)
        cls = np.asarray(dd.class_id) if getattr(dd, "class_id", None) is not None else np.zeros(len(dd.xyxy))
        ant_mask = np.zeros((H, W), bool)
        hole2 = np.zeros((H, W), bool)
        win2 = np.zeros((H, W), bool)
        for bi, b in enumerate(np.asarray(dd.xyxy)):
            bx0, by0, bx1, by1 = [int(v) for v in b]
            gbox = (cx0 + bx0, cy0 + by0, cx0 + bx1, cy0 + by1)   # crop -> image coords
            c = int(cls[bi])
            if c == ANTENNA:
                ap_prob = coarse_prob(ant, img, boxmask(gbox, H, W), SEG_HOLE, dev)
                ant_mask |= colfill(ap_prob > ANT_THR)
            elif c == HOLE:
                pr, bb = segment_box(holeseg, img, gbox, dev, HOLE_THR, SEG_REG)
                if pr is not None:
                    x0, y0, x1, y1 = bb; hole2[y0:y1, x0:x1] |= pr
            elif c == WINDOW:
                pr, bb = segment_box(winseg, img, gbox, dev, WIN_THR, SEG_REG)
                if pr is not None:
                    x0, y0, x1, y1 = bb; win2[y0:y1, x0:x1] |= pr

        # assemble final masks
        outline = keep_main(body_mask | ant_mask, 9)
        windows = tint_mask | win2                                   # tint ensemble + 2-stage
        body_hires = birefnet_outline(bfbody, img, carbox, dev, size=PUNCHOUT_SIZE)
        outline_hires = keep_main(body_hires | ant_mask, 9)
        punchout = (binary_fill_holes(outline_hires) & ~outline_hires) | hole2   # structural + 2-stage

        ours = {"outline": outline, "holes": windows, "antenna": ant_mask}
        prod = {"outline": prod_path(r, "outline"), "holes": prod_path(r, "holes_tint")}
        for c in ("outline", "holes", "antenna"):
            if gt[c].sum() < 20:
                continue
            logrow(r["image"], c, "ours", gt[c], ours[c])
            if c in prod and prod[c]:
                logrow(r["image"], c, "prod", gt[c], prodmask(prod[c]))

        np.savez_compressed(out/"masks"/f"{Path(r['image']).stem}.npz",
                            outline=outline, holes=windows, antenna=ant_mask,
                            punchout=punchout, carbox=np.array(carbox))
        if (i+1) % 25 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)
    fout.close()
    print(f"Done -> {out}/e2e_metrics.csv + masks/")


if __name__ == "__main__":
    main()
