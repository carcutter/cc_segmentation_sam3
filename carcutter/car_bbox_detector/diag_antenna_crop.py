#!/usr/bin/env python3
"""Diagnose antenna survival through the eval crop pipeline.
For each test row with a GT antenna, report:
  in_box      : antenna pixels inside the index box (x,y,bw,bh) / total antenna px
  above_box   : antenna pixels ABOVE the box top (cut by the box) / total
  surv_letter : antenna px surviving into the letterboxed model-input grid / total
  ant_h_in    : antenna height in the letterboxed grid (px)  -- thinness proxy
Also re-derive the box straight from the mask union to confirm index == union extent.
"""
import csv, os, sys
import numpy as np, cv2
from PIL import Image
sys.path.insert(0, "carcutter/car_bbox_detector/birefnet/BiRefNet")
from carcutter.car_bbox_detector.probe_birefnet_aspect import route, crop_pad_box, letterbox, BUCKETS

ROOT = "carcutter/car_bbox_detector"; WHITE = (255, 255, 255)
rows = [r for r in csv.DictReader(open(f"{ROOT}/data/index.csv")) if r["split"] == "test"]

n = 0; in_box = []; above = []; surv = []; anth = []; box_matches = 0; box_top_above_ant = 0
worst = []
for r in rows:
    rgb = np.array(Image.open(r["mask"]).convert("RGB")); H, W = rgb.shape[:2]
    ga = (rgb == WHITE).all(2)
    if ga.sum() < 20: continue
    n += 1
    tot = float(ga.sum())
    bx, by, bw, bh = int(r["x"]), int(r["y"]), int(r["bw"]), int(r["bh"])
    box = (bx, by, bx + bw, by + bh)
    # in-box fraction
    inb = ga[by:by + bh, bx:bx + bw].sum() / tot
    ays = np.where(ga.any(1))[0]; ant_top = ays.min()
    ab = ga[:by].sum() / tot                      # antenna pixels above the box top
    in_box.append(inb); above.append(ab)
    if ant_top < by: box_top_above_ant += 1       # box top is BELOW antenna tip -> cuts antenna
    # re-derive union box
    u = rgb.sum(2) > 0; uys, uxs = np.where(u)
    if (uxs.min(), uys.min()) == (bx, by): box_matches += 1
    # letterbox survival
    cx0, cy0, cx1, cy1 = crop_pad_box(box, H, W)
    bucket = route(bw / bh); BW, BH = BUCKETS[bucket]
    gac = ga[cy0:cy1, cx0:cx1].astype(np.uint8)
    lb, (ox, oy, nw, nh) = letterbox(gac, BW, BH, cv2.INTER_NEAREST)
    sv = lb.sum() / tot; surv.append(sv)
    ar = np.where(lb.any(1))[0]; anth.append((ar.max() - ar.min() + 1) if len(ar) else 0)
    worst.append((sv, os.path.basename(r["image"])[:50], inb, ab, bucket))

print(f"antenna test rows: {n}")
print(f"  index box-top matches union-extent top : {box_matches}/{n}")
print(f"  rows where box top is BELOW antenna tip (cuts it): {box_top_above_ant}/{n}")
print(f"  antenna px IN box     mean {np.mean(in_box):.3f}  min {np.min(in_box):.3f}")
print(f"  antenna px ABOVE box  mean {np.mean(above):.3f}  max {np.max(above):.3f}")
print(f"  antenna px SURVIVING letterbox  mean {np.mean(surv):.3f}  (<0.5: {sum(s<0.5 for s in surv)}/{n})")
print(f"  antenna HEIGHT in letterboxed grid (px)  mean {np.mean(anth):.1f}  median {np.median(anth):.0f}  (<8px: {sum(a<8 for a in anth)}/{n})")
worst.sort()
print("\n  lowest-survival rows (surv | inbox | abovebox | bucket | name):")
for s, nm, ib, ab, bk in worst[:12]:
    print(f"    {s:.2f}  {ib:.2f}  {ab:.2f}  {bk}  {nm}")
