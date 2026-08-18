#!/usr/bin/env python3
"""
Is small punchout-hole recall resolution-bound? Run BiRefNet body at 1024 vs 1536 on
the car crop, extract topological holes of the matte, count small-hole instance recall
vs GT (topological holes of GT union). Inference-only, no training.

Run from repo root:
  PYTHONPATH=.:carcutter/car_bbox_detector/birefnet/BiRefNet python carcutter/car_bbox_detector/hires_punchout_test.py --n 120
"""
import argparse, csv
from pathlib import Path
import numpy as np, cv2
from PIL import Image
from scipy.ndimage import binary_fill_holes
from carcutter.car_bbox_detector.birefnet_eval import load_birefnet, birefnet_outline
BF = "carcutter/car_bbox_detector/birefnet/BiRefNet/ckpts/car_outline_v2/epoch_16.pth"
MINPX = 40
SIZES = [("small", 0, 0.005), ("med", 0.005, 0.02), ("large", 0.02, 1.0)]


def comps(mask):
    n, lbl, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return [(lbl == i, int(st[i, cv2.CC_STAT_AREA])) for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= MINPX]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2e", default="carcutter/car_bbox_detector/data/e2e_v2")
    ap.add_argument("--index", default="carcutter/car_bbox_detector/data/index.csv")
    ap.add_argument("--n", type=int, default=120)
    args = ap.parse_args()
    dev = "cuda"
    idx = {Path(r["image"]).stem: r for r in csv.DictReader(open(args.index))}
    bf = load_birefnet(BF, dev)
    npzs = sorted((Path(args.e2e) / "masks").glob("*.npz"))[:args.n]
    res = {s: {sz[0]: [0, 0] for sz in SIZES} for s in (1024, 1536)}  # [gt, hit]
    prec = {1024: [0, 0], 1536: [0, 0]}
    for i, npz in enumerate(npzs):
        r = idx.get(npz.stem)
        if r is None: continue
        d = np.load(npz)
        if "carbox" not in d: continue
        img = np.array(Image.open(r["image"]).convert("RGB"))
        union = np.array(Image.open(r["mask"]).convert("RGB")).sum(2) > 0
        car = float(union.sum())
        if car < 1000: continue
        gt_holes = binary_fill_holes(union) & ~union
        def bucket(a):
            f = a / car
            for nm, lo, hi in SIZES:
                if lo <= f < hi: return nm
            return "large"
        for S in (1024, 1536):
            body = birefnet_outline(bf, img, d["carbox"], dev, size=S)
            oh = binary_fill_holes(body) & ~body
            for comp, area in comps(gt_holes):
                res[S][bucket(area)][0] += 1
                if (comp & oh).sum() >= 0.30 * area: res[S][bucket(area)][1] += 1
            for comp, area in comps(oh):
                prec[S][1] += 1
                if (comp & gt_holes).sum() >= 0.30 * area: prec[S][0] += 1
        if (i + 1) % 30 == 0: print(f"  {i+1}/{len(npzs)}", flush=True)
    print(f"\nPunchout small-hole recall by inference resolution (n={len(npzs)}):")
    for S in (1024, 1536):
        print(f"\n  size={S}:")
        tot = [0, 0]
        for nm, lo, hi in SIZES:
            g, h = res[S][nm]; tot[0] += g; tot[1] += h
            print(f"    {nm:<7} GT={g:<5} recall={100*h/g:.0f}%" if g else f"    {nm:<7} GT=0")
        print(f"    TOTAL  GT={tot[0]:<5} recall={100*tot[1]/tot[0]:.0f}%  precision={100*prec[S][0]/max(1,prec[S][1]):.0f}%")


if __name__ == "__main__":
    main()
