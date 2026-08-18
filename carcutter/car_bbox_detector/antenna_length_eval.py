#!/usr/bin/env python3
"""
THIN-ANTENNA LENGTH COVERAGE: for mast-like antennas (not blobby shark-fins), do we trace the FULL
length or just the base? Per thin GT antenna, split along its length into TIP / MID / BASE thirds and
measure coverage by our prediction (outline o, head a, and o|a) within a small tolerance band.
Viz: dimmed photo, GT mast (green, dilated), our o|a prediction (red, dilated), overlap = YELLOW.
  -> green-only = missed length; yellow = captured; red-only = over-extension.
Out: experiments/spliced_viz/antenna_length.png  + per-third coverage table.
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


def thin_metrics(gt_a):
    """return (is_thin, length, width, ys, xs). thin = elongated low-fill mast."""
    ys, xs = np.where(gt_a)
    if len(ys) < 8: return False, 0, 0, ys, xs
    h = ys.max() - ys.min() + 1; w = xs.max() - xs.min() + 1
    L = max(h, w); area = len(ys)
    fill = area / float(h * w)           # blobby fins ~0.5+, masts low
    width = area / float(L)              # mean thickness in px
    is_thin = (L >= 22) and (width <= 6.0) and (fill <= 0.35)
    return is_thin, L, width, ys, xs


def thirds_cov(gt_a, pred, ys, tol):
    """coverage of GT by dilated pred, in TIP/MID/BASE thirds along image-y (tip=smallest y=top)."""
    pd = binary_dilation(pred, iterations=tol)
    y0, y1 = ys.min(), ys.max(); edges = np.linspace(y0, y1 + 1, 4)
    out = []
    for k in range(3):
        seg = gt_a & (np.arange(gt_a.shape[0])[:, None] >= edges[k]) & (np.arange(gt_a.shape[0])[:, None] < edges[k + 1])
        s = seg.sum(); out.append((seg & pd).sum() / s if s else float("nan"))
    return out  # [tip, mid, base]


def main():
    model = Spliced(load_bn(PROD5, 1, dev), load_bn(TRI, 3, dev)).eval()
    rows = [r for r in csv.DictReader(open(f"{ROOT}/data/index.csv")) if r["split"] == "test"]
    agg = {p: defaultdict(list) for p in ("o", "a", "oa")}
    viz = []  # (tipcov_oa, crop)
    nthin = 0
    for r in rows:
        rgb = np.array(Image.open(r["mask"]).convert("RGB")); gt_a = (rgb == WHITE).all(2)
        if gt_a.sum() < 20: continue
        is_thin, L, width, ys, xs = thin_metrics(gt_a)
        if not is_thin: continue
        nthin += 1
        img = np.array(Image.open(r["image"]).convert("RGB"))
        bw, bh = int(r["bw"]), int(r["bh"]); box = (int(r["x"]), int(r["y"]), int(r["x"]) + bw, int(r["y"]) + bh)
        o, t, a = infer(model, img, box, route(bw / bh), dev)
        tol = max(2, int(round(0.03 * L)))
        oa = o | a
        for nm, pr in (("o", o), ("a", a), ("oa", oa)):
            tp, md, bs = thirds_cov(gt_a, pr, ys, tol)
            agg[nm]["tip"].append(tp); agg[nm]["mid"].append(md); agg[nm]["base"].append(bs)
        # viz crop
        pad = 30
        ry0, ry1 = max(0, ys.min() - pad), min(img.shape[0], ys.max() + pad)
        rx0, rx1 = max(0, xs.min() - pad), min(img.shape[1], xs.max() + pad)
        c = (0.4 * img[ry0:ry1, rx0:rx1]).astype(np.uint8)
        gd = binary_dilation(gt_a[ry0:ry1, rx0:rx1], iterations=2)
        pd = binary_dilation(oa[ry0:ry1, rx0:rx1], iterations=2)
        c[..., 0] = np.where(pd, 235, c[..., 0])   # R = prediction
        c[..., 1] = np.where(gd, 235, c[..., 1])   # G = GT  (overlap -> yellow)
        c = cv2.resize(c, (260, 360), interpolation=cv2.INTER_NEAREST)
        tp, md, bs = thirds_cov(gt_a, oa, ys, tol)
        cv2.putText(c, f"tip{tp*100:.0f} mid{md*100:.0f} base{bs*100:.0f}", (4, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        viz.append((tp, c))

    print(f"\nTHIN antennas (mast-like, L>=22 & width<=6 & fill<=0.35): n={nthin}")
    print(f"  coverage along mast length (tol = 3% of length), by prediction:")
    print(f"  {'pred':<6}{'TIP':>7}{'MID':>7}{'BASE':>7}")
    for nm, lab in (("o", "outline"), ("a", "head"), ("oa", "o|head")):
        d = agg[nm]
        print(f"  {lab:<6}{np.nanmean(d['tip'])*100:>6.0f}%{np.nanmean(d['mid'])*100:>6.0f}%{np.nanmean(d['base'])*100:>6.0f}%")
    print("  TIP<<BASE => we get the base but lose the thin tip. green-only in viz = missed length.")

    viz.sort(key=lambda z: z[0])  # worst tip-coverage first
    pick = viz[:7] + viz[-7:][::-1]  # 7 worst + 7 best tip
    rowsout = [np.hstack([np.hstack([c, np.full((360, 3, 3), 255, np.uint8)]) for _, c in pick[i:i+7]]) for i in range(0, len(pick), 7)]
    grid = np.vstack([np.vstack([r, np.full((3, r.shape[1], 3), 255, np.uint8)]) for r in rowsout])
    cv2.imwrite(f"{OUTD}/antenna_length.png", cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"\n  viz -> {OUTD}/antenna_length.png (top row=WORST tip-coverage, bottom=best; "
          f"GREEN=GT mast, RED=our o|head, YELLOW=overlap/captured)")


if __name__ == "__main__":
    main()
