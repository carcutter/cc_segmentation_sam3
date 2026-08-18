#!/usr/bin/env python3
"""
Montage the WORST window SOLID-miss cases: GT-tint regions we paint as opaque body (outline=1,tint=0).
Crop tight to the window region so we can judge dark-glass-miss vs mislabel. ours|prod side by side.
  green  = GT tint boundary        red = predicted tint (left ours / right prod)
Out: experiments/spliced_viz/windows_solidmiss.png
"""
import csv, os, sys
import numpy as np, cv2, torch
from PIL import Image
sys.path.insert(0, "carcutter/car_bbox_detector/birefnet/BiRefNet")
from carcutter.car_bbox_detector.seg_eval import mask_iou
from carcutter.car_bbox_detector.probe_birefnet_aspect import route
from carcutter.car_bbox_detector.build_spliced_eval import Spliced, load_bn, infer, PROD5, TRI

ROOT = "carcutter/car_bbox_detector"; OUTD = f"{ROOT}/experiments/spliced_viz"; BLUE = (0, 0, 255); dev = "cuda"


def binar(p):
    if not p or not os.path.exists(p): return None
    a = np.array(Image.open(p)); return (a.sum(2) > 0) if a.ndim == 3 else (a > 0)


def tile(img, gt, pred, roi, label):
    y0, y1, x0, x1 = roi
    c = img[y0:y1, x0:x1].copy(); g = gt[y0:y1, x0:x1]; p = pred[y0:y1, x0:x1] if pred is not None else None
    if p is not None: c[p] = (0.45 * c[p] + np.array([210, 30, 30])).clip(0, 255).astype(np.uint8)
    er = g ^ (cv2.erode(g.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0); c[er] = [0, 255, 0]
    c = cv2.resize(c, (380, 300))
    cv2.putText(c, label, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    return c


model = Spliced(load_bn(PROD5, 1, dev), load_bn(TRI, 3, dev)).eval()
rows = [r for r in csv.DictReader(open(f"{ROOT}/data/index.csv")) if r["split"] == "test"]
cand = []
for r in rows:
    rgb = np.array(Image.open(r["mask"]).convert("RGB")); gt_t = (rgb == BLUE).all(2)
    if gt_t.sum() < 600: continue
    img = np.array(Image.open(r["image"]).convert("RGB"))
    bw, bh = int(r["bw"]), int(r["bh"]); box = (int(r["x"]), int(r["y"]), int(r["x"]) + bw, int(r["y"]) + bh)
    o, t, a = infer(model, img, box, route(bw / bh), dev)
    tot = float(gt_t.sum()); fsolid = (gt_t & o & ~t).sum() / tot; iou = mask_iou(gt_t, t)
    if fsolid < 0.4: continue
    ys, xs = np.where(gt_t); pad = 50
    roi = (max(0, ys.min() - pad), min(img.shape[0], ys.max() + pad),
           max(0, xs.min() - pad), min(img.shape[1], xs.max() + pad))
    prod_t = binar(r["prod_outline"].replace("/outline/", "/holes_tint/")) if r["prod_outline"] else None
    pi = mask_iou(gt_t, prod_t) if prod_t is not None else float("nan")
    cand.append((fsolid, img, gt_t, t, prod_t, roi, iou, pi))

cand.sort(key=lambda z: -z[0])
rowsout = []
for fs, img, gt, t, pt, roi, iou, pi in cand[:12]:
    a = tile(img, gt, t, roi, f"ours IoU{iou:.2f} solid{fs:.2f}")
    b = tile(img, gt, pt, roi, f"prod IoU{pi:.2f}") if pt is not None else np.full((300, 380, 3), 40, np.uint8)
    rowsout.append(np.hstack([a, np.full((300, 4, 3), 255, np.uint8), b]))
grid = np.vstack([np.vstack([r, np.full((4, r.shape[1], 3), 255, np.uint8)]) for r in rowsout])
cv2.imwrite(f"{OUTD}/windows_solidmiss.png", cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
print(f"n candidates(solid>=0.4, tint>=600px): {len(cand)}  -> {OUTD}/windows_solidmiss.png (top 12 by solid frac)")
