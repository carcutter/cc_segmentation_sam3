#!/usr/bin/env python3
"""
Build the antenna-discriminator dataset: square crops of DETECTOR antenna boxes,
labelled pos (GT white inside >=20px = real antenna) / neg (none = stick/branch/pole).
One detector pass over train (val split held for the classifier's own val).

Out: antenna_clf_data/{train,val}/{pos,neg}/*.png

Run from repo root:
  PYTHONPATH=. python carcutter/car_bbox_detector/build_antenna_clf_data.py --n 6000
"""
import argparse, csv
from pathlib import Path
import numpy as np, cv2
from PIL import Image
import rfdetr
WHITE = (255, 255, 255); ANT = 1
EXP = "carcutter/car_bbox_detector/experiments"
OUT = "carcutter/car_bbox_detector/antenna_clf_data"


def square_crop(img, box, pad=0.20):
    H, W = img.shape[:2]
    x0, y0, x1, y1 = [float(v) for v in box]
    bw, bh = x1 - x0, y1 - y0
    px, py = bw * pad, bh * pad
    x0, y0 = max(0, int(x0 - px)), max(0, int(y0 - py))
    x1, y1 = min(W, int(x1 + px)), min(H, int(y1 + py))
    c = img[y0:y1, x0:x1]
    if c.shape[0] < 6 or c.shape[1] < 6:
        return None, None
    side = max(c.shape[:2]); xo, yo = (side - c.shape[1]) // 2, (side - c.shape[0]) // 2
    sq = np.zeros((side, side, 3), np.uint8); sq[yo:yo + c.shape[0], xo:xo + c.shape[1]] = c
    return sq, (x0, y0, x1, y1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="carcutter/car_bbox_detector/data/index.csv")
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--thr", type=float, default=0.10)
    args = ap.parse_args()
    for sp in ("train", "val"):
        for c in ("pos", "neg"):
            (Path(OUT) / sp / c).mkdir(parents=True, exist_ok=True)
    det = rfdetr.RFDETRMedium.from_checkpoint(f"{EXP}/rfdetr_car_antenna_v1/checkpoint_best_total.pth")
    rows = [r for r in csv.DictReader(open(args.index)) if r["split"] in ("train", "val")][:args.n]
    npos = nneg = 0
    for i, r in enumerate(rows):
        sp = r["split"] if r["split"] in ("train", "val") else "train"
        pil = Image.open(r["image"]).convert("RGB"); img = np.array(pil); H, W = img.shape[:2]
        white = (np.array(Image.open(r["mask"]).convert("RGB")) == WHITE).all(2)
        d = det.predict(pil, threshold=args.thr)
        cls, conf, xyxy = np.asarray(d.class_id), np.asarray(d.confidence), np.asarray(d.xyxy)
        for j in range(len(xyxy)):
            if cls[j] != ANT or conf[j] < args.thr:
                continue
            x0, y0, x1, y1 = [int(v) for v in xyxy[j]]
            x0, y0 = max(0, x0), max(0, y0); x1, y1 = min(W, x1), min(H, y1)
            wbox = int(white[y0:y1, x0:x1].sum())
            sq, _ = square_crop(img, xyxy[j])
            if sq is None:
                continue
            lab = "pos" if wbox >= 20 else "neg"
            stem = f"{Path(r['image']).stem}_b{j}"
            cv2.imwrite(str(Path(OUT) / sp / lab / f"{stem}.png"), cv2.cvtColor(sq, cv2.COLOR_RGB2BGR))
            if lab == "pos":
                npos += 1
            else:
                nneg += 1
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(rows)}  pos={npos} neg={nneg}", flush=True)
    print(f"done: pos={npos} neg={nneg}")
    for sp in ("train", "val"):
        print(f"  {sp}: pos={len(list((Path(OUT)/sp/'pos').glob('*.png')))} "
              f"neg={len(list((Path(OUT)/sp/'neg').glob('*.png')))}")


if __name__ == "__main__":
    main()
