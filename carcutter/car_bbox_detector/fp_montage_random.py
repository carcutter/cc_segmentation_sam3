#!/usr/bin/env python3
"""FAST (cache): UNBIASED random sample of non-wheel small-hole FP components + size distribution.
Unlike fp_montage.py (top-30 by area, biased to big see-through), this samples uniformly so the
montage represents the FP population. Out: experiments/spliced_viz/nonwheel_hole_fp_random.png"""
import os, random
import numpy as np, cv2
from PIL import Image
from scipy.ndimage import binary_fill_holes
from carcutter.car_bbox_detector.e2e_cache_lib import iter_cached, wheel_region, ROOT

OUTD = f"{ROOT}/experiments/spliced_viz"; MINPX = 25
SIZE_BINS = [("tiny", 25, 100), ("small", 100, 400), ("med", 400, 1500), ("big", 1500, 1e9)]  # px area


def comps(mask, ca):
    n, lbl, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return [(lbl == i, int(st[i, cv2.CC_STAT_AREA])) for i in range(1, n)
            if MINPX <= st[i, cv2.CC_STAT_AREA] and st[i, cv2.CC_STAT_AREA] / ca < 0.005]


def main():
    fps = []  # (stem, image_path, (ry0,ry1,rx0,rx1), area_px)
    sizehist = {b[0]: 0 for b in SIZE_BINS}
    for r, rgb, pred in iter_cached():
        gt_o = rgb.sum(2) > 0
        if gt_o.sum() < 1500: continue
        ca = float(gt_o.sum()); gt_h = binary_fill_holes(gt_o) & ~gt_o; wreg = wheel_region(rgb)
        H, W = gt_o.shape
        for comp, area in comps(pred["punchout"], ca):
            if (comp & wreg).sum() >= 0.5 * area: continue
            if (comp & gt_h).sum() >= 0.3 * area: continue
            ys, xs = np.where(comp); pad = 55
            fps.append((r["image"], (max(0,ys.min()-pad), min(H,ys.max()+pad), max(0,xs.min()-pad), min(W,xs.max()+pad)),
                        ys.min(), ys.max(), xs.min(), xs.max(), area))
            for nm, lo, hi in SIZE_BINS:
                if lo <= area < hi: sizehist[nm] += 1

    print(f"\nTotal non-wheel small-hole FP components: {len(fps)}")
    print("  size distribution (px area):  " + "  ".join(f"{nm}:{sizehist[nm]} ({100*sizehist[nm]/max(1,len(fps)):.0f}%)" for nm,_,_ in SIZE_BINS))
    rng = random.Random(0); samp = rng.sample(fps, min(30, len(fps)))
    tiles = []
    for imgpath, (ry0,ry1,rx0,rx1), y0,y1,x0,x1, area in samp:
        img = np.array(Image.open(imgpath).convert("RGB"))
        cr = img[ry0:ry1, rx0:rx1].copy()
        # re-mark the FP comp region (recompute locally cheap: just box it, exact mask not stored)
        cv2.rectangle(cr, (x0-rx0, y0-ry0), (x1-rx0, y1-ry0), (235,30,30), 2)
        cr = cv2.resize(cr, (260, 260))
        cv2.putText(cr, f"{area}px {os.path.basename(imgpath)[:12]}", (4,18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)
        tiles.append(cr)
    rows_ = [np.hstack([np.hstack([t, np.full((260,3,3),255,np.uint8)]) for t in tiles[i:i+6]]) for i in range(0,len(tiles),6)]
    w = max(r.shape[1] for r in rows_); rows_ = [np.hstack([r, np.full((260,w-r.shape[1],3),30,np.uint8)]) if r.shape[1]<w else r for r in rows_]
    grid = np.vstack([np.vstack([r, np.full((3,w,3),255,np.uint8)]) for r in rows_])
    cv2.imwrite(f"{OUTD}/nonwheel_hole_fp_random.png", cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"  random {len(tiles)} -> {OUTD}/nonwheel_hole_fp_random.png (RED BOX = FP location)")


if __name__ == "__main__":
    main()
