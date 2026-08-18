#!/usr/bin/env python3
"""
Window-hole (tint) ensemble: does UNet ∪ BiRefNet-tint beat either alone? Different
architectures -> different misses -> union should lift recall (esp small/dark-glass
windows). Reports IoU + instance recall by window size, vs prod holes_tint.

unet = cached e2e_v2 d['holes']; birefnet-tint = run ckpt on the car crop.
Run from repo root:
  PYTHONPATH=.:carcutter/car_bbox_detector/birefnet/BiRefNet \
  python carcutter/car_bbox_detector/windows_ensemble_eval.py --ckpt <tint epoch.pth>
"""
import argparse, csv, os
from pathlib import Path
import numpy as np, cv2
from PIL import Image
from carcutter.car_bbox_detector.birefnet_eval import load_birefnet, birefnet_outline
from carcutter.car_bbox_detector.seg_eval import mask_iou
BLUE = (0, 0, 255); MINPX = 40
SIZES = [("small", 0, 0.005), ("med", 0.005, 0.02), ("large", 0.02, 1.0)]


def binar(p):
    if not p or not os.path.exists(p): return None
    a = np.array(Image.open(p)); return (a.sum(2) > 0) if a.ndim == 3 else (a > 0)


def comps(mask):
    n, lbl, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return [(lbl == i, int(st[i, cv2.CC_STAT_AREA])) for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= MINPX]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--e2e", default="carcutter/car_bbox_detector/data/e2e_v2")
    ap.add_argument("--index", default="carcutter/car_bbox_detector/data/index.csv")
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()
    dev = "cuda"
    idx = {Path(r["image"]).stem: r for r in csv.DictReader(open(args.index))}
    tint = load_birefnet(args.ckpt, dev)
    methods = ["unet", "birefnet", "union", "inter", "prod"]
    iou = {m: [] for m in methods}
    car_area_frac = {}
    rec = {m: {s[0]: [0, 0] for s in SIZES} for m in methods}   # by GT-window size
    npzs = sorted((Path(args.e2e) / "masks").glob("*.npz"))[:args.n]
    for i, npz in enumerate(npzs):
        r = idx.get(npz.stem)
        if r is None: continue
        d = np.load(npz)
        if "carbox" not in d: continue
        img = np.array(Image.open(r["image"]).convert("RGB"))
        rgb = np.array(Image.open(r["mask"]).convert("RGB")); blue = (rgb == BLUE).all(2)
        if blue.sum() < 50: continue
        car = float((rgb.sum(2) > 0).sum())
        unet = d["holes"].astype(bool)
        bf = birefnet_outline(tint, img, d["carbox"], dev)     # tint model on car crop
        po = r["prod_outline"]; prod = binar(po.replace("/outline/", "/holes_tint/")) if (po and os.path.exists(po)) else None
        masks = {"unet": unet, "birefnet": bf, "union": unet | bf, "inter": unet & bf, "prod": prod}
        for m, mk in masks.items():
            if mk is None: continue
            iou[m].append(mask_iou(blue, mk))
        def bucket(a):
            f = a / car
            for nm, lo, hi in SIZES:
                if lo <= f < hi: return nm
            return "large"
        for comp, area in comps(blue):
            b = bucket(area)
            for m, mk in masks.items():
                if mk is None: continue
                rec[m][b][0] += 1
                if (comp & mk).sum() >= 0.30 * area: rec[m][b][1] += 1
        if (i + 1) % 50 == 0: print(f"  {i+1}/{len(npzs)}", flush=True)

    mn = lambda a: float(np.mean(a)) if a else float("nan")
    print(f"\nWindow (tint) ensemble, n={len(npzs)} (ckpt={Path(args.ckpt).name}):\n")
    print(f"  {'method':<9}{'IoU':>8}   instance recall by GT window size (small/med/large/TOTAL)")
    for m in methods:
        if not iou[m]: continue
        tot = [0, 0]; cells = []
        for nm, _, _ in SIZES:
            g, h = rec[m][nm]; tot[0] += g; tot[1] += h
            cells.append(f"{nm} {100*h/g:.0f}%" if g else f"{nm} -")
        cells.append(f"TOT {100*tot[1]/tot[0]:.0f}%" if tot[0] else "-")
        print(f"  {m:<9}{mn(iou[m]):>8.3f}   " + "  ".join(cells))


if __name__ == "__main__":
    main()
