#!/usr/bin/env python3
"""
Validate the bet: do SMALL punchout holes live in a few predictable regions?
Small holes = topological holes of the GT car silhouette with area < SMALL_FRAC of car
(wheel/bumper/step-up/roof-rail gaps). Build a per-view body-frame heatmap of where they
occur + report concentration. If tight -> region-proposal + dedicated small-hole model is viable.

Run from repo root:
  PYTHONPATH=. python carcutter/car_bbox_detector/small_hole_locality.py
"""
import argparse, csv, re
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes
import cv2
from scipy.ndimage import gaussian_filter
U0, U1, V0, V1, GW, GH = -0.1, 1.1, -0.1, 1.1, 80, 80
SMALL_FRAC = 0.005
MINPX = 30


def view_of(p):
    m = re.search(r'-(front-left|front-right|rear-left|rear-right|side-left|side-right|front|rear|side)\.jpg$', p)
    return m.group(1) if m else "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="carcutter/car_bbox_detector/data/index.csv")
    ap.add_argument("--out", default="carcutter/car_bbox_detector/data/small_hole_locality.png")
    args = ap.parse_args()
    rows = [r for r in csv.DictReader(open(args.index)) if r["split"] in ("train", "val")]
    acc = {}; cnt = {}; n_holes = 0; n_img = 0
    for r in rows:
        union = np.array(Image.open(r["mask"]).convert("RGB")).sum(2) > 0
        ca = float(union.sum())
        if ca < 1000: continue
        ys, xs = np.where(union); bx0, by0 = xs.min(), ys.min()
        bw, bh = max(1, xs.max() - bx0), max(1, ys.max() - by0)
        holes = (binary_fill_holes(union) & ~union).astype(np.uint8)
        nlab, lbl, st, cent = cv2.connectedComponentsWithStats(holes, 8)
        view = view_of(r["image"]); n_img += 1
        h = np.zeros((GH, GW), np.float32); got = False
        for i in range(1, nlab):
            a = st[i, cv2.CC_STAT_AREA]
            if a < MINPX or a / ca >= SMALL_FRAC: continue   # small holes only
            cx, cy = cent[i]
            u = (cx - bx0) / bw; v = (cy - by0) / bh
            gi = int((u - U0) / (U1 - U0) * GW); gj = int((v - V0) / (V1 - V0) * GH)
            if 0 <= gi < GW and 0 <= gj < GH:
                h[gj, gi] += 1; got = True; n_holes += 1
        if got:
            acc.setdefault(view, np.zeros((GH, GW), np.float32)); acc[view] += h
            cnt[view] = cnt.get(view, 0) + 1
    print(f"small holes found: {n_holes} across {n_img} imgs; per-view img counts: {sorted(cnt.items(), key=lambda x:-x[1])}")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    views = sorted(acc, key=lambda v: -cnt[v])
    pooled = sum(acc.values())
    maps = {v: gaussian_filter(acc[v], 2) for v in views}; maps["all"] = gaussian_filter(pooled, 2)
    order = ["all"] + views; n = len(order); cols = 5; rows_ = (n + cols - 1) // cols
    fig, ax = plt.subplots(rows_, cols, figsize=(4 * cols, 4 * rows_)); ax = np.atleast_2d(ax)
    for i, v in enumerate(order):
        m = maps[v] / (maps[v].max() + 1e-9)
        a = ax[i // cols, i % cols]; a.imshow(m, cmap="inferno", aspect="auto")
        hot = 100 * (m > 0.1).mean()
        a.set_title(f"{v} (hot={hot:.0f}%)", fontsize=10); a.set_xticks([]); a.set_yticks([])
    for j in range(n, rows_ * cols): ax[j // cols, j % cols].axis("off")
    fig.suptitle("SMALL punchout-hole locality (body frame; top=roof, bottom=wheels/bumper)", fontsize=13)
    fig.tight_layout(); fig.savefig(args.out, dpi=85, bbox_inches="tight"); print(f"-> {args.out}")


if __name__ == "__main__":
    main()
