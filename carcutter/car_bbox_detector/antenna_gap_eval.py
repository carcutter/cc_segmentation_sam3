#!/usr/bin/env python3
"""
Is our thin-antenna mask DISCONTINUOUS (internal gaps -> joinable with a line) vs just losing the tip
(trailing loss -> nothing to join)? Per thin mast, walk tip->base over GT rows, mark covered[y] = GT@y
within tol of our prediction. Classify:
  internal-gap  : a 0-run flanked by 1s (covered..gap..covered) = FRAGMENTED, fixable by collinear join.
  tip-loss only : uncovered only at the leading (tip) end = NOT joinable.
Report fraction in each class + mean #gaps + mean gap length, for pred = head(a) and o|head.
Viz fragmented cases: antenna_gaps.png (green=GT, red=pred, yellow=overlap; tight tol).
"""
import csv, os, sys
from collections import defaultdict
import numpy as np, cv2, torch
from PIL import Image
from scipy.ndimage import binary_dilation
sys.path.insert(0, "carcutter/car_bbox_detector/birefnet/BiRefNet")
from carcutter.car_bbox_detector.probe_birefnet_aspect import route
from carcutter.car_bbox_detector.build_spliced_eval import Spliced, load_bn, infer, PROD5, TRI

ROOT = "carcutter/car_bbox_detector"; OUTD = f"{ROOT}/experiments/spliced_viz"; WHITE = (255, 255, 255); dev = "cuda"


def thin(gt_a):
    ys, xs = np.where(gt_a)
    if len(ys) < 8: return False, ys, xs
    h = ys.max() - ys.min() + 1; w = xs.max() - xs.min() + 1; L = max(h, w)
    return (L >= 22) and (len(ys) / L <= 6.0) and (len(ys) / float(h * w) <= 0.35), ys, xs


def classify(gt_a, pred, ys, tol):
    """returns (n_internal_gaps, internal_gap_px, tiploss_px, covered_frac) walking tip(min y)->base."""
    pd = binary_dilation(pred, iterations=tol)
    y0, y1 = ys.min(), ys.max()
    seq = []  # covered per row that has GT
    for y in range(y0, y1 + 1):
        g = gt_a[y]
        if not g.any(): continue
        seq.append(bool((g & pd[y]).any()))
    seq = np.array(seq)
    if seq.size == 0: return 0, 0, 0, 0.0
    cov = seq.mean()
    # leading (tip) uncovered run
    lead = 0
    for v in seq:
        if v: break
        lead += 1
    # internal gaps = 0-runs that are flanked by 1s on BOTH sides
    ones = np.where(seq)[0]
    n_gap, gap_px = 0, 0
    if len(ones) >= 2:
        for a, b in zip(ones[:-1], ones[1:]):
            if b - a > 1:
                n_gap += 1; gap_px += (b - a - 1)
    return n_gap, gap_px, lead, cov


def main():
    model = Spliced(load_bn(PROD5, 1, dev), load_bn(TRI, 3, dev)).eval()
    rows = [r for r in csv.DictReader(open(f"{ROOT}/data/index.csv")) if r["split"] == "test"]
    stat = {p: {"frag": 0, "tiponly": 0, "perfect": 0, "ngaps": [], "gappx": [], "n": 0} for p in ("a", "oa")}
    viz = []
    for r in rows:
        rgb = np.array(Image.open(r["mask"]).convert("RGB")); gt_a = (rgb == WHITE).all(2)
        if gt_a.sum() < 20: continue
        ist, ys, xs = thin(gt_a)
        if not ist: continue
        img = np.array(Image.open(r["image"]).convert("RGB"))
        bw, bh = int(r["bw"]), int(r["bh"]); box = (int(r["x"]), int(r["y"]), int(r["x"]) + bw, int(r["y"]) + bh)
        o, t, a = infer(model, img, box, route(bw / bh), dev)
        L = max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1); tol = max(1, int(round(0.02 * L)))
        for nm, pr in (("a", a), ("oa", o | a)):
            ng, gpx, lead, cov = classify(gt_a, pr, ys, tol)
            s = stat[nm]; s["n"] += 1; s["ngaps"].append(ng); s["gappx"].append(gpx)
            if ng > 0: s["frag"] += 1
            elif lead > 0: s["tiponly"] += 1
            else: s["perfect"] += 1
        ng_oa = classify(gt_a, o | a, ys, tol)[0]
        if ng_oa > 0:  # collect fragmented for viz
            pad = 26; ry0, ry1 = max(0, ys.min()-pad), min(img.shape[0], ys.max()+pad)
            rx0, rx1 = max(0, xs.min()-pad), min(img.shape[1], xs.max()+pad)
            c = (0.4 * img[ry0:ry1, rx0:rx1]).astype(np.uint8)
            gd = binary_dilation(gt_a[ry0:ry1, rx0:rx1], iterations=2); pdv = binary_dilation((o|a)[ry0:ry1, rx0:rx1], iterations=tol)
            c[..., 0] = np.where(pdv, 235, c[..., 0]); c[..., 1] = np.where(gd, 235, c[..., 1])
            c = cv2.resize(c, (240, 360), interpolation=cv2.INTER_NEAREST)
            cv2.putText(c, f"{ng_oa}gap", (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
            viz.append((ng_oa, c))

    print(f"\nTHIN-MAST CONTINUITY (n={stat['a']['n']} masts, tol=2% length):")
    print(f"  {'pred':<7}{'FRAGMENTED':>12}{'tip-loss only':>15}{'fully covered':>15}{'mean#gaps':>11}{'mean gap px':>12}")
    for nm, lab in (("a", "head"), ("oa", "o|head")):
        s = stat[nm]; n = max(1, s["n"])
        print(f"  {lab:<7}{100*s['frag']/n:>10.0f}% {100*s['tiponly']/n:>13.0f}% {100*s['perfect']/n:>13.0f}%"
              f"{np.mean(s['ngaps']):>11.2f}{np.mean(s['gappx']):>12.1f}")
    print("  FRAGMENTED = internal gap (covered..gap..covered) = JOINABLE by collinear bridge.")
    print("  tip-loss only = uncovered at tip end = NOT joinable.")

    if viz:
        viz.sort(key=lambda z: -z[0]); pick = viz[:21]
        rr = [np.hstack([np.hstack([c, np.full((360,3,3),255,np.uint8)]) for _,c in pick[i:i+7]]) for i in range(0,len(pick),7)]
        w = max(r.shape[1] for r in rr); rr = [np.hstack([r, np.full((360, w-r.shape[1],3),30,np.uint8)]) if r.shape[1]<w else r for r in rr]
        cv2.imwrite(f"{OUTD}/antenna_gaps.png", cv2.cvtColor(np.vstack([np.vstack([r,np.full((3,w,3),255,np.uint8)]) for r in rr]), cv2.COLOR_RGB2BGR))
        print(f"\n  {len(viz)} fragmented masts -> {OUTD}/antenna_gaps.png (green=GT, red=pred o|head, yellow=overlap)")


if __name__ == "__main__":
    main()
