#!/usr/bin/env python3
"""
Precision-recovery crops: train the hole/window segmenters on the DETECTOR'S OWN boxes
(train/deploy parity) instead of GT-region boxes. Runs the unified multi-class detector on
each train/val car crop, then for every predicted box crops EXACTLY that box (no pad, exactly
as eval's segment_box does) -> resize 512, label = GT mask within the box. Detector FP boxes
(no GT inside) become empty-label HARD NEGATIVES that teach the segmenter to stay silent.

class 1 holeregion  -> holeseg_detframed/   label = topological small holes
class 2 windowregion-> windowseg_detframed/ label = tint (blue)

Single-process (CUDA). Run from repo root:
  PYTHONPATH=. python carcutter/car_bbox_detector/build_detframed_seg_crops.py --det-thr 0.3
"""
import argparse, csv
from pathlib import Path
import numpy as np, cv2
import rfdetr
from PIL import Image
from scipy.ndimage import binary_fill_holes
BLUE = (0, 0, 255)
SMALL_FRAC = 0.006; H_MINPX = 25; SZ = 512; CROP_PAD = 0.15
HOLE_DIR = "carcutter/car_bbox_detector/holeseg_detframed"
WIN_DIR = "carcutter/car_bbox_detector/windowseg_detframed"


def small_holes(union, ca):
    holes = (binary_fill_holes(union) & ~union).astype(np.uint8)
    n, lbl, st, _ = cv2.connectedComponentsWithStats(holes, 8)
    out = np.zeros_like(holes)
    for i in range(1, n):
        if H_MINPX <= st[i, cv2.CC_STAT_AREA] < SMALL_FRAC * ca:
            out[lbl == i] = 1
    return out.astype(bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default="carcutter/car_bbox_detector/experiments/rfdetr_multiclass_v1/checkpoint_best_ema.pth")
    ap.add_argument("--index", default="carcutter/car_bbox_detector/data/index.csv")
    ap.add_argument("--det-thr", type=float, default=0.3)
    args = ap.parse_args()
    for base in (HOLE_DIR, WIN_DIR):
        for sp in ("train", "val"):
            for d in ("images", "labels"):
                Path(f"{base}/{sp}/{d}").mkdir(parents=True, exist_ok=True)
    det = rfdetr.RFDETRMedium.from_checkpoint(args.det)
    rows = [r for r in csv.DictReader(open(args.index)) if r["split"] in ("train", "val")]
    cnt = {"hole": [0, 0], "win": [0, 0]}   # [total, with-label]
    for ri, r in enumerate(rows):
        rgb = np.array(Image.open(r["mask"]).convert("RGB")); union = rgb.sum(2) > 0
        ca = float(union.sum())
        if ca < 1500: continue
        img = np.array(Image.open(r["image"]).convert("RGB")); H, W = img.shape[:2]
        x, y, bw, bh = int(r["x"]), int(r["y"]), int(r["bw"]), int(r["bh"])
        px, py = int(bw * CROP_PAD), int(bh * CROP_PAD)
        cx0, cy0 = max(0, x - px), max(0, y - py); cx1, cy1 = min(W, x + bw + px), min(H, y + bh + py)
        carcrop = img[cy0:cy1, cx0:cx1]
        if carcrop.shape[0] < 32 or carcrop.shape[1] < 32: continue
        sh = small_holes(union, ca); blue = (rgb == BLUE).all(2)
        dd = det.predict(Image.fromarray(carcrop), threshold=args.det_thr)
        cls = np.asarray(dd.class_id) if getattr(dd, "class_id", None) is not None else np.zeros(len(dd.xyxy))
        stem = Path(r["image"]).stem; sp = r["split"]
        for bi, b in enumerate(np.asarray(dd.xyxy)):
            c = int(cls[bi])
            if c == 1: base, gt, key = HOLE_DIR, sh, "hole"
            elif c == 2: base, gt, key = WIN_DIR, blue, "win"
            else: continue
            bx0, by0, bx1, by1 = [int(v) for v in b]            # crop coords -> native
            gx0, gy0, gx1, gy1 = cx0 + bx0, cy0 + by0, cx0 + bx1, cy0 + by1
            gx0, gy0 = max(0, gx0), max(0, gy0); gx1, gy1 = min(W, gx1), min(H, gy1)
            if gx1 - gx0 < 12 or gy1 - gy0 < 12: continue
            im = cv2.resize(img[gy0:gy1, gx0:gx1], (SZ, SZ))
            lb = cv2.resize((gt[gy0:gy1, gx0:gx1] * 255).astype(np.uint8), (SZ, SZ), interpolation=cv2.INTER_NEAREST)
            nm = f"{stem}__b{bi}"
            cv2.imwrite(f"{base}/{sp}/images/{nm}.png", cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
            cv2.imwrite(f"{base}/{sp}/labels/{nm}.png", lb)
            cnt[key][0] += 1; cnt[key][1] += int(lb.max() > 0)
        if ri % 1000 == 0:
            print(f"  {ri}/{len(rows)}  hole {cnt['hole'][0]}({cnt['hole'][1]} pos)  win {cnt['win'][0]}({cnt['win'][1]} pos)", flush=True)
    print(f"DONE  hole crops {cnt['hole'][0]} ({cnt['hole'][1]} with holes)  |  win crops {cnt['win'][0]} ({cnt['win'][1]} with windows)")


if __name__ == "__main__":
    main()
