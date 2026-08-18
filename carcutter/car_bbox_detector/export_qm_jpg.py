#!/usr/bin/env python3
"""
Export the QM Exterior results as flat JPGs — what the quality manager actually asked for, after
Photoshop refused the PSDs. No GPU: reads the masks run_qm_eval.py already wrote.

  ours_jpg/      the final cutout composited on white, at panel-1 resolution and filename
  overlay_jpg/   the untouched panel-1 photo with our cut line drawn on it (the over-crop check that
                 the hidden ORIGINAL layer used to provide inside the PSD)

Same stem as the input, so the QM can stitch RAW | prod | ours by filename.

Run: python carcutter/car_bbox_detector/export_qm_jpg.py [--n 4] [--bg 255]
"""
import argparse
import glob
import os
from multiprocessing import Pool

import cv2
import numpy as np
from PIL import Image

SET = "/home/rutger/work/cc_segmentation_sam3/data/car_segmentation/qm_eval_202608_car_segmentation"
QUALITY = 95


def work(a):
    src, out, bg, contour = a
    stem = os.path.splitext(os.path.basename(src))[0]
    img = np.array(Image.open(src).convert("RGB"))
    H, W = img.shape[:2]
    o = f"{SET}/ours/outline/{stem}.png"
    if not os.path.exists(o):
        return ("nomask", stem)
    outline = np.array(Image.open(o).convert("L")) > 127
    pth = f"{SET}/ours/punchout/{stem}.png"
    alpha = outline.copy()
    if os.path.exists(pth):
        alpha &= ~(np.array(Image.open(pth).convert("L")) > 127)

    plate = np.full_like(img, bg)
    a3 = alpha[..., None].astype(np.float32)
    cut = (img * a3 + plate * (1 - a3)).astype(np.uint8)
    Image.fromarray(cut).save(f"{out}/ours_jpg/{stem}.jpg", quality=QUALITY, subsampling=0)

    if contour:
        viz = img.copy()
        m = outline.astype(np.uint8)
        k = max(2, round(min(W, H) / 500))
        edge = cv2.dilate(m, np.ones((k, k), np.uint8)) - cv2.erode(m, np.ones((k, k), np.uint8))
        viz[edge > 0] = (0, 255, 0)
        Image.fromarray(viz).save(f"{out}/overlay_jpg/{stem}.jpg", quality=QUALITY)
    return ("ok", stem)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--out", default=SET)
    ap.add_argument("--bg", type=int, default=255, help="background grey level for the cutout JPG")
    ap.add_argument("--no-contour", action="store_true")
    args = ap.parse_args()
    for d in ("ours_jpg", "overlay_jpg"):
        os.makedirs(f"{args.out}/{d}", exist_ok=True)
    srcs = sorted(glob.glob(f"{SET}/input/*.jpg"))
    if args.n:
        srcs = srcs[:args.n]
    with Pool(12) as p:
        res = p.map(work, [(s, args.out, args.bg, not args.no_contour) for s in srcs], chunksize=4)
    bad = [s for st, s in res if st != "ok"]
    print(f"exported {len(res)-len(bad)}/{len(res)} -> {args.out}/ours_jpg (+ overlay_jpg)")
    for s in bad:
        print("  MISSING MASK:", s)


if __name__ == "__main__":
    main()
