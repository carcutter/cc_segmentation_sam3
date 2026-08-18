#!/usr/bin/env python3
"""FAST (from cache): render worst NON-WHEEL small-hole FP cases. Run after cache_pipeline.py.
Out: experiments/spliced_viz/nonwheel_hole_fp.png"""
import os, heapq
import numpy as np, cv2
from PIL import Image
from scipy.ndimage import binary_fill_holes
from carcutter.car_bbox_detector.e2e_cache_lib import iter_cached, wheel_region, ROOT

OUTD = f"{ROOT}/experiments/spliced_viz"; MINPX = 25; KEEP = 30


def comps(mask, ca, lo=0, hi=1.0):
    n, lbl, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return [(lbl == i, st[i, cv2.CC_STAT_AREA]) for i in range(1, n)
            if MINPX <= st[i, cv2.CC_STAT_AREA] and lo <= st[i, cv2.CC_STAT_AREA] / ca < hi]


def main():
    heap = []; uid = 0
    for r, rgb, pred in iter_cached():
        gt_o = rgb.sum(2) > 0
        if gt_o.sum() < 1500: continue
        ca = float(gt_o.sum()); gt_h = binary_fill_holes(gt_o) & ~gt_o; wreg = wheel_region(rgb)
        img = np.array(Image.open(r["image"]).convert("RGB")); H, W = img.shape[:2]
        for comp, area in comps(pred["punchout"], ca, 0, 0.005):
            if (comp & wreg).sum() >= 0.5 * area: continue
            if (comp & gt_h).sum() >= 0.3 * area: continue
            ys, xs = np.where(comp); pad = 55
            ry0, ry1 = max(0, ys.min()-pad), min(H, ys.max()+pad); rx0, rx1 = max(0, xs.min()-pad), min(W, xs.max()+pad)
            cr = img[ry0:ry1, rx0:rx1].copy(); cm = comp[ry0:ry1, rx0:rx1]; gh = gt_h[ry0:ry1, rx0:rx1]
            cr[cm] = (0.4*cr[cm] + np.array([235,30,30])).clip(0,255).astype(np.uint8)
            ge = gh ^ (cv2.erode(gh.astype(np.uint8), np.ones((3,3),np.uint8)) > 0); cr[ge] = [0,255,0]
            cr = cv2.resize(cr, (260, 260))
            cv2.putText(cr, f"{1000*area/ca:.1f}/k {os.path.basename(r['image'])[:14]}", (4,18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)
            uid += 1
            if len(heap) < KEEP: heapq.heappush(heap, (area/ca, uid, cr))
            elif area/ca > heap[0][0]: heapq.heappushpop(heap, (area/ca, uid, cr))
    tiles = [c for _, _, c in sorted(heap, key=lambda z: -z[0])]
    rows_ = [np.hstack([np.hstack([t, np.full((260,3,3),255,np.uint8)]) for t in tiles[i:i+6]]) for i in range(0, len(tiles), 6)]
    w = max(r.shape[1] for r in rows_); rows_ = [np.hstack([r, np.full((260, w-r.shape[1],3),30,np.uint8)]) if r.shape[1]<w else r for r in rows_]
    grid = np.vstack([np.vstack([r, np.full((3,w,3),255,np.uint8)]) for r in rows_])
    cv2.imwrite(f"{OUTD}/nonwheel_hole_fp.png", cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"{len(tiles)} worst non-wheel hole FPs -> {OUTD}/nonwheel_hole_fp.png (RED=spurious hole, green=real GT holes)")


if __name__ == "__main__":
    main()
