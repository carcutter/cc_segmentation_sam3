#!/usr/bin/env python3
"""
Rebuild the BiRefNet training set to fix the two segment-drop failure modes:
  - OPEN-TRUNK/tailgate drops: oversample 'trunk-straight' views (x3)
  - detector-clip drops: larger crop pad (0.12 -> 0.15)
CAR-TR = all trunk-straight train (x oversample) + N random non-trunk; CAR-VD = all val.
Crops = union-bbox + pad, letterboxed square (no distortion). Clears existing CAR-TR/CAR-VD.

Run from repo root:
  PYTHONPATH=. python carcutter/car_bbox_detector/build_birefnet_retrain.py
"""
import argparse, csv, random, shutil
from pathlib import Path
import numpy as np, cv2
from PIL import Image
ROOT = "carcutter/car_bbox_detector/birefnet/data/datasets/dis/General"


def letterbox(a, interp):
    h, w = a.shape[:2]; s = max(h, w); xo, yo = (s - w) // 2, (s - h) // 2
    sq = np.zeros((s, s, 3), a.dtype) if a.ndim == 3 else np.zeros((s, s), a.dtype)
    sq[yo:yo + h, xo:xo + w] = a; return sq


def make_crop(r, pad):
    rgb = np.array(Image.open(r["mask"]).convert("RGB"))
    gt = rgb.sum(2) > 0
    if gt.sum() < 30: return None
    img = np.array(Image.open(r["image"]).convert("RGB")); H, W = img.shape[:2]
    x, y, bw, bh = int(r["x"]), int(r["y"]), int(r["bw"]), int(r["bh"])
    px, py = int(bw * pad), int(bh * pad)
    x0, y0 = max(0, x - px), max(0, y - py); x1, y1 = min(W, x + bw + px), min(H, y + bh + py)
    im = letterbox(img[y0:y1, x0:x1], cv2.INTER_LINEAR)
    gm = letterbox((gt[y0:y1, x0:x1].astype(np.uint8) * 255), cv2.INTER_NEAREST)
    return im, gm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="carcutter/car_bbox_detector/data/index.csv")
    ap.add_argument("--pad", type=float, default=0.15)
    ap.add_argument("--trunk-oversample", type=int, default=3)
    ap.add_argument("--n-other", type=int, default=3200)
    args = ap.parse_args()
    for sub in ("CAR-TR", "CAR-VD"):
        for d in ("im", "gt"):
            p = Path(ROOT) / sub / d
            if p.exists(): shutil.rmtree(p)
            p.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(args.index)))
    tr = [r for r in rows if r["split"] == "train"]
    va = [r for r in rows if r["split"] == "val"]
    trunk = [r for r in tr if "trunk-straight" in r["image"]]
    other = [r for r in tr if "trunk-straight" not in r["image"]]
    random.seed(7); random.shuffle(other); other = other[:args.n_other]
    print(f"trunk train={len(trunk)} (x{args.trunk_oversample}) + other={len(other)} | val={len(va)} | pad={args.pad}")

    def write(r, sub, tag=""):
        c = make_crop(r, args.pad)
        if c is None: return 0
        im, gm = c; stem = Path(r["image"]).stem + tag
        cv2.imwrite(f"{ROOT}/{sub}/im/{stem}.jpg", cv2.cvtColor(im, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(f"{ROOT}/{sub}/gt/{stem}.png", gm); return 1
    n = 0
    for r in other: n += write(r, "CAR-TR")
    for r in trunk:
        for k in range(args.trunk_oversample): n += write(r, "CAR-TR", f"_o{k}")
    nv = sum(write(r, "CAR-VD") for r in va)
    print(f"CAR-TR={n}  CAR-VD={nv}")


if __name__ == "__main__":
    main()
