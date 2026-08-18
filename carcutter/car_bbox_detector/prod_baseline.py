#!/usr/bin/env python3
"""
Baseline: how well does the PRODUCTION outline model's bbox match the GT
car-extent bbox? This is the bar to beat.

Coverage-first metrics (a clipped car is the expensive error):
  - IoU(GT, prod)
  - top_clip   = max(0, prod_top - gt_top)      px the prod box cuts off the TOP
  - any_clip   = max over the 4 edges of how far prod sits *inside* GT (px)
  - contained  = prod box fully contains GT box (no clipping at all)
Broken out for ALL test images vs the ANTENNA subset (the hard, important case).

Usage:
    python prod_baseline.py                       # uses data/index.csv test split
    python prod_baseline.py --split test
"""

import argparse
import csv
import numpy as np
from PIL import Image


def prod_bbox(path):
    a = np.array(Image.open(path))
    b = a > 0 if a.ndim == 2 else a.sum(2) > 0
    ys, xs = np.where(b)
    if len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]  # xyxy


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def summarize(label, recs):
    if not recs:
        print(f"  {label}: (none)"); return
    iouv = np.array([r["iou"] for r in recs])
    topc = np.array([r["top_clip"] for r in recs])
    anyc = np.array([r["any_clip"] for r in recs])
    cont = np.array([r["contained"] for r in recs])
    print(f"  {label} (n={len(recs)}):")
    print(f"    IoU         mean {iouv.mean():.3f}  median {np.median(iouv):.3f}")
    print(f"    top_clip px mean {topc.mean():5.1f}  p90 {np.percentile(topc,90):5.1f}  max {topc.max():.0f}  (% clipped >2px: {100*(topc>2).mean():.1f}%)")
    print(f"    any_clip px mean {anyc.mean():5.1f}  p90 {np.percentile(anyc,90):5.1f}")
    print(f"    fully contains GT: {100*cont.mean():.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/index.csv")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    recs, missing = [], 0
    for r in csv.DictReader(open(args.index)):
        if r["split"] != args.split or not r["prod_outline"]:
            continue
        pb = prod_bbox(r["prod_outline"])
        if pb is None:
            missing += 1; continue
        gx, gy, gw, gh = int(r["x"]), int(r["y"]), int(r["bw"]), int(r["bh"])
        gt = [gx, gy, gx + gw, gy + gh]
        recs.append({
            "iou": iou(gt, pb),
            "top_clip": max(0, pb[1] - gt[1]),
            "any_clip": max(pb[0]-gt[0], pb[1]-gt[1], gt[2]-pb[2], gt[3]-pb[3], 0),
            "contained": int(pb[0] <= gt[0] and pb[1] <= gt[1] and pb[2] >= gt[2] and pb[3] >= gt[3]),
            "antenna": int(r["has_antenna"]),
        })
    print(f"PROD baseline on '{args.split}' (n={len(recs)}, {missing} prod masks empty/missing)\n")
    summarize("ALL", recs)
    summarize("ANTENNA subset", [r for r in recs if r["antenna"]])
    summarize("non-antenna", [r for r in recs if not r["antenna"]])


if __name__ == "__main__":
    main()
