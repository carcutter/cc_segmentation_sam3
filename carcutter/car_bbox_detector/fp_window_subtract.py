#!/usr/bin/env python3
"""FAST (cache): how much does subtracting the WINDOW (tint) mask from punchout fix the small-hole FP?
Compares small-hole recall/precision (wheel/nonwheel) for punchout vs punchout & ~dilate(GT-blue).
GT-blue = clean upper bound on the glass-FP portion. Run after cache_pipeline.py."""
import numpy as np, cv2
from scipy.ndimage import binary_fill_holes, binary_dilation
from carcutter.car_bbox_detector.e2e_cache_lib import iter_cached, wheel_region

BLUE = (0, 0, 255); MINPX = 25


def comps(mask, ca, lo, hi):
    n, lbl, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return [(lbl == i, st[i, cv2.CC_STAT_AREA]) for i in range(1, n)
            if MINPX <= st[i, cv2.CC_STAT_AREA] and lo <= st[i, cv2.CC_STAT_AREA] / ca < hi]


def stats(gt_h, pred, ca, region):
    g = [(c, a) for c, a in comps(gt_h, ca, 0, 0.005) if (c & region).sum() >= 0.5 * a]
    p = [(c, a) for c, a in comps(pred, ca, 0, 0.005) if (c & region).sum() >= 0.5 * a]
    tp = sum(1 for c, a in g if (c & pred).sum() >= 0.3 * a)
    fp = sum(1 for c, a in p if (c & gt_h).sum() < 0.3 * a)
    return len(g), tp, fp


def main():
    acc = {m: {r: [0, 0, 0] for r in ("wheel", "nonwheel")} for m in ("raw", "minus_win")}
    for r, rgb, pred in iter_cached():
        gt_o = rgb.sum(2) > 0
        if gt_o.sum() < 1500: continue
        ca = float(gt_o.sum()); gt_h = binary_fill_holes(gt_o) & ~gt_o
        wreg = wheel_region(rgb); nwreg = ~wreg
        win = binary_dilation((rgb == BLUE).all(2), iterations=3)
        po = pred["punchout"]; po_mw = po & ~win
        for mname, p in (("raw", po), ("minus_win", po_mw)):
            for rname, reg in (("wheel", wreg), ("nonwheel", nwreg)):
                for k, v in zip(range(3), stats(gt_h, p, ca, reg)): acc[mname][rname][k] += v
    print(f"\nSMALL-hole punchout, effect of subtracting WINDOW(GT-blue) mask:")
    print(f"  {'variant':<10}{'region':<9}{'GT#':>6}{'recall':>8}{'prec':>7}{'FP':>6}")
    for m in ("raw", "minus_win"):
        for rg in ("wheel", "nonwheel"):
            g, tp, fp = acc[m][rg]
            print(f"  {m:<10}{rg:<9}{g:>6}{(100*tp/g if g else 0):>7.0f}%{(100*tp/max(1,tp+fp)):>6.0f}%{fp:>6}")
    print("  (recall unchanged => window-subtract only removes FPs that were on labeled glass)")


if __name__ == "__main__":
    main()
