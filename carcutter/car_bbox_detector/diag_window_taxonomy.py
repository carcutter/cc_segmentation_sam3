#!/usr/bin/env python3
"""
Decompose GT-tint (blue) regions by what OUR model thinks they are, to separate two failure modes:
  TINT       : (gt_t & tint_pred)            -- we agree it's see-through glass
  SEETHROUGH : (gt_t & ~outline & ~tint)     -- we carved it out of the silhouette (open/rolled-down
                                                window -> our PUNCHOUT-HOLE channel). Taxonomy disagreement,
                                                arguably MORE correct than prod labeling open glass as tint.
  SOLID      : (gt_t & outline & ~tint)       -- we painted it as opaque car body = a REAL miss (dark glass).
Reported overall, and split for the low-tint-IoU "loser" rows (IoU<0.3) which dominate windows_lose.
"""
import csv, os, sys
from collections import defaultdict
import numpy as np, cv2, torch
from PIL import Image
from scipy.ndimage import binary_fill_holes
sys.path.insert(0, "carcutter/car_bbox_detector/birefnet/BiRefNet")
from carcutter.car_bbox_detector.seg_eval import mask_iou
from carcutter.car_bbox_detector.probe_birefnet_aspect import route
from carcutter.car_bbox_detector.build_spliced_eval import Spliced, load_bn, infer, PROD5, TRI

ROOT = "carcutter/car_bbox_detector"; BLUE = (0, 0, 255); dev = "cuda"
model = Spliced(load_bn(PROD5, 1, dev), load_bn(TRI, 3, dev)).eval()
rows = [r for r in csv.DictReader(open(f"{ROOT}/data/index.csv")) if r["split"] == "test"]

agg = {"all": defaultdict(list), "lose": defaultdict(list)}
n = 0
for r in rows:
    rgb = np.array(Image.open(r["mask"]).convert("RGB"))
    gt_t = (rgb == BLUE).all(2)
    if gt_t.sum() < 60: continue
    n += 1
    img = np.array(Image.open(r["image"]).convert("RGB"))
    bw, bh = int(r["bw"]), int(r["bh"]); box = (int(r["x"]), int(r["y"]), int(r["x"]) + bw, int(r["y"]) + bh)
    o, t, a = infer(model, img, box, route(bw / bh), dev)
    tot = float(gt_t.sum()); iou = mask_iou(gt_t, t)
    holes = binary_fill_holes(o) & ~o
    f_tint = (gt_t & t).sum() / tot
    f_solid = (gt_t & o & ~t).sum() / tot
    f_see = (gt_t & ~o & ~t).sum() / tot          # carved out of silhouette (open/rolled-down)
    f_hole = (gt_t & holes).sum() / tot            # specifically enclosed punchout-hole
    bucket = "lose" if iou < 0.3 else "all"
    for b in ("all", bucket if bucket == "lose" else "all"):
        agg[b]["iou"].append(iou); agg[b]["tint"].append(f_tint)
        agg[b]["solid"].append(f_solid); agg[b]["see"].append(f_see); agg[b]["hole"].append(f_hole)

def show(name, d):
    if not d["iou"]: return
    print(f"  {name:<6} n={len(d['iou']):<4} IoU {np.mean(d['iou']):.2f} | "
          f"as TINT {np.mean(d['tint']):.2f} | SEETHROUGH(open) {np.mean(d['see']):.2f} "
          f"(enclosed-hole {np.mean(d['hole']):.2f}) | SOLID-miss {np.mean(d['solid']):.2f}")

print(f"\nGT-tint rows: {n}  (fractions are share of GT-blue pixels)")
print("  How OUR model explains the GT-tint region:")
show("ALL", agg["all"])
show("LOSE", agg["lose"])  # IoU<0.3
print("\n  SEETHROUGH high in LOSE => taxonomy disagreement (open windows labeled tint), not a real miss.")
print("  SOLID high => genuine dark-glass miss.")
