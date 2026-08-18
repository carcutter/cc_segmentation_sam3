#!/usr/bin/env python3
"""
Mine antenna FALSE-POSITIVE crops as hard negatives.

The detector proposes antenna boxes at ~0.5 precision by design (recall-first).
Every antenna box with NO GT white inside is a stick-like FP (pole/cable/bar/etc.).
Saving those box crops with an all-zero label teaches the antenna UNet "stick != antenna",
which connectivity cleanup can't fix (these sticks connect to the car).

Writes hard_negatives/{images,labels}/<id>.png (square box crop + zero mask),
mirroring the deploy antenna crop (square-padded detector box).

Run from repo root:
  PYTHONPATH=. python carcutter/car_bbox_detector/antenna_mine_fp.py --n 3500
"""
import argparse, csv
from pathlib import Path
import numpy as np, cv2
from PIL import Image
import rfdetr
WHITE = (255, 255, 255)
ANT = 1
EXP = "carcutter/car_bbox_detector/experiments"


def square_crop(img, box, pad=0.15):
    H, W = img.shape[:2]
    x0, y0, x1, y1 = [float(v) for v in box]
    bw, bh = x1 - x0, y1 - y0
    px, py = bw * pad, bh * pad
    x0, y0 = max(0, int(x0 - px)), max(0, int(y0 - py))
    x1, y1 = min(W, int(x1 + px)), min(H, int(y1 + py))
    crop = img[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    if ch < 4 or cw < 4:
        return None
    side = max(ch, cw); xo, yo = (side - cw) // 2, (side - ch) // 2
    sq = np.zeros((side, side, 3), np.uint8); sq[yo:yo + ch, xo:xo + cw] = crop
    return sq, (x0, y0, x1, y1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="carcutter/car_bbox_detector/data/index.csv")
    ap.add_argument("--out", default="carcutter/car_bbox_detector/antenna_hard_negs")
    ap.add_argument("--n", type=int, default=3500)
    ap.add_argument("--thr", type=float, default=0.15, help="antenna detection threshold")
    args = ap.parse_args()
    out = Path(args.out); (out / "images").mkdir(parents=True, exist_ok=True); (out / "labels").mkdir(parents=True, exist_ok=True)
    det = rfdetr.RFDETRMedium.from_checkpoint(f"{EXP}/rfdetr_car_antenna_v1/checkpoint_best_total.pth")
    rows = [r for r in csv.DictReader(open(args.index)) if r["split"] == "train"][:args.n]

    saved = 0
    for i, r in enumerate(rows):
        pil = Image.open(r["image"]).convert("RGB"); img = np.array(pil); H, W = img.shape[:2]
        white = (np.array(Image.open(r["mask"]).convert("RGB")) == WHITE).all(2)
        d = det.predict(pil, threshold=args.thr)
        cls, conf, xyxy = np.asarray(d.class_id), np.asarray(d.confidence), np.asarray(d.xyxy)
        for j in range(len(xyxy)):
            if cls[j] != ANT or conf[j] < args.thr:
                continue
            x0, y0, x1, y1 = [int(v) for v in xyxy[j]]
            x0, y0 = max(0, x0), max(0, y0); x1, y1 = min(W, x1), min(H, y1)
            wbox = white[y0:y1, x0:x1].sum()
            if wbox >= 20:           # real antenna in box -> not a negative
                continue
            res = square_crop(img, xyxy[j])
            if res is None:
                continue
            sq, _ = res
            stem = f"{Path(r['image']).stem}_fp{j}"
            cv2.imwrite(str(out / "images" / f"{stem}.png"), cv2.cvtColor(sq, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out / "labels" / f"{stem}.png"), np.zeros(sq.shape[:2], np.uint8))
            saved += 1
        if (i + 1) % 250 == 0:
            print(f"  {i+1}/{len(rows)}  FP crops so far: {saved}", flush=True)
    print(f"mined {saved} antenna FP hard-negative crops -> {out}")


if __name__ == "__main__":
    main()
