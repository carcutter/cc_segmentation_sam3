#!/usr/bin/env python3
"""
Build the BiRefNet fine-tuning dataset: CAR-BOX crops (not raw frames) so the car
fills the 1024 input. Crop = union bbox (from index) + pad, letterboxed to square
(no aspect distortion), saved into BiRefNet's layout:
  {root}/datasets/dis/General/CAR-TR/{im,gt}/   (train)
  {root}/datasets/dis/General/CAR-VD/{im,gt}/   (val)
im = RGB jpg, gt = 0/255 union mask png (same crop). Deploy mirrors this: crop the
RF-DETR detector box, letterbox, resize 1024.

Run from repo root:
  /home/rutger/miniconda3/envs/cc_sam3/bin/python carcutter/car_bbox_detector/build_birefnet_data.py --target union
"""
import argparse, csv, os
from pathlib import Path
from multiprocessing import Pool
import numpy as np, cv2
from PIL import Image

BLUE, WHITE = (0, 0, 255), (255, 255, 255)
ROOT = "carcutter/car_bbox_detector/birefnet/data/datasets/dis/General"  # overridden by --out-root
PAD = 0.12  # fraction of bbox size added each side (emulates detector slack)


def target_mask(rgb, target):
    if target == "union":  return rgb.sum(2) > 0
    if target == "blue":   return (rgb == BLUE).all(2)
    if target == "white":  return (rgb == WHITE).all(2)


def crop_box(H, W, x, y, bw, bh):
    px, py = int(bw * PAD), int(bh * PAD)
    x0, y0 = max(0, x - px), max(0, y - py)
    x1, y1 = min(W, x + bw + px), min(H, y + bh + py)
    return x0, y0, x1, y1


def letterbox(arr, interp):
    h, w = arr.shape[:2]
    side = max(h, w)
    xo, yo = (side - w) // 2, (side - h) // 2
    if arr.ndim == 3:
        sq = np.zeros((side, side, 3), arr.dtype)
    else:
        sq = np.zeros((side, side), arr.dtype)
    sq[yo:yo + h, xo:xo + w] = arr
    return sq


def work(args_tuple):
    r, target = args_tuple
    split = r["split"]
    sub = {"train": "CAR-TR", "val": "CAR-VD"}.get(split)
    if sub is None:
        return 0  # skip test (held out for final eval)
    stem = Path(r["image"]).stem
    im_dst = Path(ROOT) / sub / "im" / f"{stem}.jpg"
    gt_dst = Path(ROOT) / sub / "gt" / f"{stem}.png"
    if im_dst.exists() and gt_dst.exists():
        return 0
    try:
        rgb = np.array(Image.open(r["mask"]).convert("RGB"))
        gt = target_mask(rgb, target)
        if gt.sum() < 30:
            return 0
        img = np.array(Image.open(r["image"]).convert("RGB"))
        H, W = img.shape[:2]
        x, y, bw, bh = int(r["x"]), int(r["y"]), int(r["bw"]), int(r["bh"])
        x0, y0, x1, y1 = crop_box(H, W, x, y, bw, bh)
        im_c = letterbox(img[y0:y1, x0:x1], cv2.INTER_LINEAR)
        gt_c = letterbox((gt[y0:y1, x0:x1].astype(np.uint8) * 255), cv2.INTER_NEAREST)
        cv2.imwrite(str(im_dst), cv2.cvtColor(im_c, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(gt_dst), gt_c)
        return 1
    except Exception as e:
        print("ERR", stem, e)
        return 0


def main():
    global ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="carcutter/car_bbox_detector/data/index.csv")
    ap.add_argument("--target", default="union", choices=["union", "blue", "white"])
    ap.add_argument("--out-root", default=ROOT, help="BiRefNet General dataset dir to write CAR-TR/CAR-VD into")
    args = ap.parse_args()
    ROOT = args.out_root
    for sub in ("CAR-TR", "CAR-VD"):
        for d in ("im", "gt"):
            (Path(ROOT) / sub / d).mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(args.index)))
    with Pool(16) as p:
        res = p.map(work, [(r, args.target) for r in rows], chunksize=16)
    for sub in ("CAR-TR", "CAR-VD"):
        n = len(list((Path(ROOT) / sub / "im").glob("*.jpg")))
        print(f"  {sub}: {n} images")
    print(f"wrote {sum(res)} new crops")


if __name__ == "__main__":
    main()
