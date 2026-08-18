#!/usr/bin/env python3
"""
Can we drop the BiRefNet-tint (Swin-L) pass? Measure windows = 2-stage-ALONE (unified detector
windowregion -> window-seg v2 UNet, NO tint) vs union(tint∪2stage) vs prod, on pixel IoU + boundary-F
against GT tint (blue). Also renders best-wins / worst-failures of ours(2stage) vs prod.

Run from repo root:
  PYTHONPATH=. /home/rutger/miniconda3/envs/cc_sam3/bin/python carcutter/car_bbox_detector/eval_window_drop_tint.py --n 200
"""
import argparse, csv, os
from pathlib import Path
import numpy as np, cv2, torch
from PIL import Image
import segmentation_models_pytorch as smp
import rfdetr
from carcutter.car_bbox_detector.seg_eval import boundary_f, mask_iou, MEAN, STD
from carcutter.car_bbox_detector.cascade_eval import predict_union
from carcutter.car_bbox_detector.birefnet_eval import load_birefnet, birefnet_outline

ROOT = "carcutter/car_bbox_detector"; EXP = f"{ROOT}/experiments"
TINT_CKPT = f"{ROOT}/birefnet/BiRefNet/ckpts/car_holes_v2/epoch_16.pth"
BLUE = (0, 0, 255); WINDOW = 2; DET_THR = 0.5; WIN_THR = 0.6; SZ = 512


def load_unet(ckpt, enc, dev):
    m = smp.Unet(enc, encoder_weights=None, in_channels=3, classes=1).to(dev).eval()
    st = torch.load(ckpt, map_location=dev); m.load_state_dict(st.get("model", st)); return m


@torch.no_grad()
def segment_box(seg, img, box, dev, thr):
    H, W = img.shape[:2]; x0, y0, x1, y1 = [int(v) for v in box]
    x0, y0 = max(0, x0), max(0, y0); x1, y1 = min(W, x1), min(H, y1)
    if x1 - x0 < 8 or y1 - y0 < 8: return None, None
    tile = cv2.resize(img[y0:y1, x0:x1], (SZ, SZ))
    t = torch.from_numpy(((tile / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]).float().to(dev)
    pr = (torch.sigmoid(seg(t))[0, 0] > thr).cpu().numpy().astype(np.uint8)
    pr = cv2.resize(pr, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST) > 0
    return pr, (x0, y0, x1, y1)


def overlay(img, gt, pred, box):
    """Crop to car box; GT outline in green, pred fill in red (overlap -> yellow-ish)."""
    x0, y0, x1, y1 = [int(v) for v in box]
    c = img[y0:y1, x0:x1].copy()
    g = gt[y0:y1, x0:x1]; p = pred[y0:y1, x0:x1]
    c[p] = (0.45 * c[p] + np.array([200, 0, 0])).clip(0, 255).astype(np.uint8)      # pred red
    er = g ^ cv2.erode(g.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)  # GT boundary
    c[er] = [0, 255, 0]
    return cv2.resize(c, (320, 320))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--index", default=f"{ROOT}/data/index.csv")
    ap.add_argument("--out", default=f"{ROOT}/data/window_drop_tint"); args = ap.parse_args()
    dev = "cuda"; outd = Path(args.out); outd.mkdir(parents=True, exist_ok=True)
    cardet = rfdetr.RFDETRMedium.from_checkpoint(f"{EXP}/rfdetr_car_antenna_v1/checkpoint_best_total.pth")
    unidet = rfdetr.RFDETRMedium.from_checkpoint(f"{EXP}/rfdetr_multiclass_v1/checkpoint_best_ema.pth")
    bftint = load_birefnet(TINT_CKPT, dev)
    winseg = load_unet(f"{EXP}/unet_windowseg_v2/best.pt", "efficientnet-b2", dev)
    rows = [r for r in csv.DictReader(open(args.index)) if r["split"] == "test"][:args.n]

    agg = {"2stage": [], "union": [], "prod": []}
    aggb = {"2stage": [], "union": [], "prod": []}
    recs = []   # (delta_iou, image, box, gt, ours2s, prodm, iou2s, iouprod)
    for i, r in enumerate(rows):
        pil = Image.open(r["image"]).convert("RGB"); img = np.array(pil); H, W = img.shape[:2]
        rgb = np.array(Image.open(r["mask"]).convert("RGB")); gt = (rgb == BLUE).all(2)
        if gt.sum() < 200: continue
        carbox = predict_union(cardet, pil, 0.3, 0.15)
        if carbox is None: continue
        cx0, cy0, cx1, cy1 = [max(0, int(v)) for v in carbox]; cx1, cy1 = min(W, cx1), min(H, cy1)
        if cx1 - cx0 < 32 or cy1 - cy0 < 32: continue
        carcrop = img[cy0:cy1, cx0:cx1]
        win2 = np.zeros((H, W), bool)
        dd = unidet.predict(Image.fromarray(carcrop), threshold=DET_THR)
        cls = np.asarray(dd.class_id) if getattr(dd, "class_id", None) is not None else np.zeros(len(dd.xyxy))
        for bi, b in enumerate(np.asarray(dd.xyxy)):
            if int(cls[bi]) != WINDOW: continue
            bx0, by0, bx1, by1 = [int(v) for v in b]
            pr, bb = segment_box(winseg, img, (cx0 + bx0, cy0 + by0, cx0 + bx1, cy0 + by1), dev, WIN_THR)
            if pr is not None: x0, y0, x1, y1 = bb; win2[y0:y1, x0:x1] |= pr
        tint = birefnet_outline(bftint, img, carbox, dev)
        union = tint | win2
        prodp = r["prod_outline"].replace("/outline/", "/holes_tint/") if r["prod_outline"] else ""
        prodm = (np.array(Image.open(prodp)).sum(2) > 0 if np.array(Image.open(prodp)).ndim == 3
                 else np.array(Image.open(prodp)) > 0) if (prodp and os.path.exists(prodp)) else None
        i2, iu = mask_iou(gt, win2), mask_iou(gt, union)
        agg["2stage"].append(i2); agg["union"].append(iu)
        aggb["2stage"].append(boundary_f(gt, win2, 2)); aggb["union"].append(boundary_f(gt, union, 2))
        if prodm is not None:
            ip = mask_iou(gt, prodm); agg["prod"].append(ip); aggb["prod"].append(boundary_f(gt, prodm, 2))
            recs.append((i2 - ip, r["image"], (cx0, cy0, cx1, cy1), gt, win2, prodm, i2, ip))
        if (i + 1) % 25 == 0: print(f"  {i+1}/{len(rows)}", flush=True)

    print("\n=== windows pixel quality (mean over {} imgs w/ GT windows) ===".format(len(agg["2stage"])))
    print(f"  {'variant':<10}{'IoU':>8}{'BF@2':>8}{'n':>6}")
    for k in ("2stage", "union", "prod"):
        n = len(agg[k]); print(f"  {k:<10}{np.mean(agg[k]):>8.3f}{np.mean(aggb[k]):>8.3f}{n:>6}")
    print(f"\n  drop-tint cost: IoU {np.mean(agg['2stage'])-np.mean(agg['union']):+.3f}, "
          f"vs prod {np.mean(agg['2stage'])-np.mean(agg['prod']):+.3f}")

    # render best wins (delta>0) and worst failures (delta<0) of ours(2stage) vs prod
    recs.sort(key=lambda x: x[0])
    def montage(items, path, title):
        tiles = []
        for d, im, box, g, ours, prodm, i2, ip in items:
            o = overlay(np.array(Image.open(im).convert("RGB")), g, ours, box)
            p = overlay(np.array(Image.open(im).convert("RGB")), g, prodm, box)
            cv2.putText(o, f"ours {i2:.2f}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(p, f"prod {ip:.2f}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            tiles.append(np.hstack([o, np.full((320, 4, 3), 255, np.uint8), p]))
        grid = np.vstack([np.hstack([t, np.full((320, 4, 3), 255, np.uint8)]) for t in []] or
                         [np.vstack([tiles[k], np.full((4, tiles[k].shape[1], 3), 255, np.uint8)]) for k in range(len(tiles))])
        cv2.imwrite(str(path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"  {title}: {path}")
    montage(recs[:8], outd / "worst_vs_prod.png", "worst failures (prod beats ours)")
    montage(list(reversed(recs))[:8], outd / "best_vs_prod.png", "best wins (ours beats prod)")
    print("  (green = GT window boundary, red = prediction)")


if __name__ == "__main__":
    main()
