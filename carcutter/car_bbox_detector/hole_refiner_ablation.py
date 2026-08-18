#!/usr/bin/env python3
"""FAST (cache): ablate the 2-stage hole-UNet refiner. Compare small-hole recall/FP (wheel/nonwheel)
for punchout WITHOUT refiner (BiRefNet silhouette structural holes only) vs WITH (|hole2), and report
exactly what the refiner ADDS (recall gain, FP gain, and the refiner's own added-component precision)."""
import numpy as np, cv2
from scipy.ndimage import binary_fill_holes
from carcutter.car_bbox_detector.e2e_cache_lib import iter_cached, wheel_region
from carcutter.car_bbox_detector.antenna_fp_clean import keep_main

MINPX = 25


def comps(mask, ca):
    n, lbl, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return [(lbl == i, int(st[i, cv2.CC_STAT_AREA])) for i in range(1, n)
            if MINPX <= st[i, cv2.CC_STAT_AREA] and st[i, cv2.CC_STAT_AREA] / ca < 0.005]


def stats(gt_h, pred, ca, region):
    g = [(c, a) for c, a in comps(gt_h, ca) if (c & region).sum() >= 0.5 * a]
    p = [(c, a) for c, a in comps(pred, ca) if (c & region).sum() >= 0.5 * a]
    tp = sum(1 for c, a in g if (c & pred).sum() >= 0.3 * a)
    fp = sum(1 for c, a in p if (c & gt_h).sum() < 0.3 * a)
    return len(g), tp, fp


def main():
    acc = {m: {r: [0,0,0] for r in ("wheel","nonwheel")} for m in ("norefiner","refiner")}
    add = {"tp":0,"fp":0}  # refiner-added components (in hole2, not in structural) classified vs GT
    for r, rgb, pred in iter_cached():
        gt_o = rgb.sum(2) > 0
        if gt_o.sum() < 1500: continue
        ca = float(gt_o.sum()); gt_h = binary_fill_holes(gt_o) & ~gt_o
        wreg = wheel_region(rgb); nwreg = ~wreg
        outline = keep_main(pred["o"] | pred["ant"], 9)
        structural = binary_fill_holes(outline) & ~outline
        with_ref = structural | pred["hole2"]
        for mname, p in (("norefiner", structural), ("refiner", with_ref)):
            for rname, reg in (("wheel", wreg), ("nonwheel", nwreg)):
                for k, v in zip(range(3), stats(gt_h, p, ca, reg)): acc[mname][rname][k] += v
        # refiner-ADDED components = hole2 regions not already covered by structural
        added = pred["hole2"] & ~binary_fill_holes(structural | structural)  # = hole2 & ~structural roughly
        added = pred["hole2"] & ~structural
        for c, a in comps(added, ca):
            add["tp" if (c & gt_h).sum() >= 0.3*a else "fp"] += 1
    print("\nHOLE-REFINER ABLATION (small punchout holes, instance):")
    print(f"  {'variant':<10}{'region':<9}{'GT#':>6}{'recall':>8}{'prec':>7}{'FP':>6}")
    for m in ("norefiner","refiner"):
        for rg in ("wheel","nonwheel"):
            g,tp,fp = acc[m][rg]
            print(f"  {m:<10}{rg:<9}{g:>6}{(100*tp/g if g else 0):>7.0f}%{(100*tp/max(1,tp+fp)):>6.0f}%{fp:>6}")
    print(f"\n  refiner-ADDED components (in hole2, not in silhouette): TP {add['tp']}  FP {add['fp']}"
          f"  -> {100*add['tp']/max(1,add['tp']+add['fp']):.0f}% land on a labeled GT hole")
    print("  (but 'FP' here is vs the UNDER-LABELED GT — many are real rail/step-bar see-through)")


if __name__ == "__main__":
    main()
