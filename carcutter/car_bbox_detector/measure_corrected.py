#!/usr/bin/env python3
"""
APPLES-TO-APPLES re-measure of the spliced model vs REAL prod, fixing two biases:

 BUG 1 (outline): our outline predicts punchout holes CARVED OUT (trained on gt_o = rgb.sum>0,
   which excludes the internal holes). Prod's raw outline mask is a SOLID silhouette (holes are a
   SEPARATE prod model). So comparing ours(hole-carved) vs prod_raw(hole-filled) penalizes prod at
   every hole boundary. Fair prod outline = prod_outline & ~prod_holes_punchout.
   -> report ours / prod_raw / prod_corrected.

 BUG 2 (antenna): gt_o = rgb.sum>0 INCLUDES the white antenna pixels, so the OUTLINE itself already
   grabs antenna -- for both prod and us. Standalone antenna IoU is meaningless. Report, over GT-antenna
   pixels: coverage by prod outline, by our outline, by (our outline | our antenna head), and the
   MARGINAL gain the head adds on top of our own outline. Dump a few example crops.

Run: PYTHONPATH=.:.../BiRefNet HF_HOME=... PY measure_corrected.py --n 930
"""
import argparse, csv, os, sys
from collections import defaultdict
import numpy as np, cv2, torch
from PIL import Image
from scipy.ndimage import binary_fill_holes
sys.path.insert(0, "carcutter/car_bbox_detector/birefnet/BiRefNet")
from carcutter.car_bbox_detector.seg_eval import boundary_f, mask_iou
from carcutter.car_bbox_detector.probe_birefnet_aspect import route
from carcutter.car_bbox_detector.build_spliced_eval import Spliced, load_bn, infer, vg, PROD5, TRI

ROOT = "carcutter/car_bbox_detector"; EXP = f"{ROOT}/experiments"; OUTD = f"{EXP}/spliced_viz"
BLUE = (0, 0, 255); WHITE = (255, 255, 255)
V = ["straight", "corner", "corner34", "side", "other", "ALL"]


def binar(p):
    if not p or not os.path.exists(p): return None
    a = np.array(Image.open(p)); return (a.sum(2) > 0) if a.ndim == 3 else (a > 0)


def antenna_crop(img, box, gt_a, o, a, lo):
    """side-by-side crop: GT antenna green, our-outline coverage blue, head-only-extra red."""
    x0, y0, x1, y1 = [int(v) for v in box]
    ys, xs = np.where(gt_a)
    if len(ys) == 0: return None
    # tight antenna roi padded
    ay0, ay1, ax0, ax1 = ys.min(), ys.max(), xs.min(), xs.max()
    pad = max(20, (ay1 - ay0) // 2, (ax1 - ax0) // 2)
    ry0, ry1 = max(0, ay0 - pad), min(img.shape[0], ay1 + pad)
    rx0, rx1 = max(0, ax0 - pad), min(img.shape[1], ax1 + pad)
    c = img[ry0:ry1, rx0:rx1].copy()
    g = gt_a[ry0:ry1, rx0:rx1]; oo = o[ry0:ry1, rx0:rx1]; aa = a[ry0:ry1, rx0:rx1]
    head_extra = aa & ~oo            # what the head adds beyond outline
    c[oo & g] = (0.4 * c[oo & g] + np.array([40, 80, 230])).clip(0, 255).astype(np.uint8)   # outline-covered (blue)
    c[head_extra & g] = (0.3 * c[head_extra & g] + np.array([230, 40, 40])).clip(0, 255).astype(np.uint8)  # head-extra (red)
    er = g ^ (cv2.erode(g.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    c[er] = [0, 255, 0]               # GT antenna boundary (green)
    c = cv2.resize(c, (300, 300), interpolation=cv2.INTER_NEAREST)
    cv2.putText(c, lo, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
    return c


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=930)
    ap.add_argument("--index", default=f"{ROOT}/data/index.csv"); args = ap.parse_args(); dev = "cuda"
    os.makedirs(OUTD, exist_ok=True)
    model = Spliced(load_bn(PROD5, 1, dev), load_bn(TRI, 3, dev)).eval()
    rows = [r for r in csv.DictReader(open(args.index)) if r["split"] == "test"][:args.n]

    ob = {k: defaultdict(list) for k in ("ours", "prod_raw", "prod_corr")}
    # antenna coverage accumulators (by view): fractions over gt_a pixels
    AC = defaultdict(lambda: {"prod_o": [], "our_o": [], "our_oa": [], "marg": [], "head_iou": []})
    ant_examples = []  # (marg, crop)

    for i, r in enumerate(rows):
        img = np.array(Image.open(r["image"]).convert("RGB"))
        rgb = np.array(Image.open(r["mask"]).convert("RGB"))
        gt_o = rgb.sum(2) > 0
        if gt_o.sum() < 1500: continue
        gt_a = (rgb == WHITE).all(2)
        bw, bh = int(r["bw"]), int(r["bh"]); box = (int(r["x"]), int(r["y"]), int(r["x"]) + bw, int(r["y"]) + bh)
        v = vg(os.path.basename(r["image"])); po = r["prod_outline"]
        o, t, a = infer(model, img, box, route(bw / bh), dev)
        prod_o = binar(po); prod_h = binar(po.replace("/outline/", "/holes_punchout/")) if po else None
        if prod_o is not None:
            prod_corr = prod_o & ~prod_h if prod_h is not None else prod_o
            for vv in (v, "ALL"):
                ob["ours"][vv].append(boundary_f(gt_o, o, 1))
                ob["prod_raw"][vv].append(boundary_f(gt_o, prod_o, 1))
                ob["prod_corr"][vv].append(boundary_f(gt_o, prod_corr, 1))
        if gt_a.sum() >= 20:
            ga = float(gt_a.sum())
            cov_prod = (gt_a & prod_o).sum() / ga if prod_o is not None else float("nan")
            cov_o = (gt_a & o).sum() / ga
            cov_oa = (gt_a & (o | a)).sum() / ga
            for vv in (v, "ALL"):
                AC[vv]["prod_o"].append(cov_prod); AC[vv]["our_o"].append(cov_o)
                AC[vv]["our_oa"].append(cov_oa); AC[vv]["marg"].append(cov_oa - cov_o)
                AC[vv]["head_iou"].append(mask_iou(gt_a, a))
            if len(ant_examples) < 400:
                cr = antenna_crop(img, box, gt_a, o, a, f"+head {100*(cov_oa-cov_o):.0f}% (o {100*cov_o:.0f}%)")
                if cr is not None: ant_examples.append((cov_oa - cov_o, cr))
        del img, rgb, o, t, a, prod_o, prod_h
        if (i + 1) % 150 == 0: print(f"  {i+1}/{len(rows)}", flush=True)

    print("\n=== OUTLINE BF@1 (apples-to-apples: prod_corr = prod_outline & ~prod_holes) ===")
    print(f"  {'view':<9}{'n':>5}{'OURS':>8}{'PROD_raw':>10}{'PROD_corr':>11}{'gain vs corr':>14}")
    for v in V:
        if ob["ours"][v]:
            mo, mr, mc = (np.mean(ob[k][v]) for k in ("ours", "prod_raw", "prod_corr"))
            print(f"  {v:<9}{len(ob['ours'][v]):>5}{mo:>8.3f}{mr:>10.3f}{mc:>11.3f}{mo-mc:>+14.3f}")

    print("\n=== ANTENNA: coverage of GT-antenna pixels (does outline already grab it? does head add?) ===")
    print(f"  {'view':<9}{'n':>5}{'prod_outl':>10}{'our_outl':>10}{'our+head':>10}{'HEAD_adds':>11}{'head_IoU':>10}")
    for v in V:
        if AC[v]["our_o"]:
            d = AC[v]
            print(f"  {v:<9}{len(d['our_o']):>5}{np.nanmean(d['prod_o']):>10.2f}{np.mean(d['our_o']):>10.2f}"
                  f"{np.mean(d['our_oa']):>10.2f}{np.mean(d['marg']):>+11.2f}{np.mean(d['head_iou']):>10.2f}")
    print("  (prod_outl/our_outl/our+head = fraction of GT antenna pixels covered; HEAD_adds = marginal recall the head contributes over our own outline)")

    # dump antenna examples where the head adds the MOST and where it adds ~nothing
    ant_examples.sort(key=lambda z: z[0])
    def grid(items, path):
        tiles = [c for _, c in items]
        rows_ = [np.hstack([np.hstack([t, np.full((300, 3, 3), 255, np.uint8)]) for t in tiles[i:i+5]])
                 for i in range(0, len(tiles), 5)]
        w = max(r.shape[1] for r in rows_)
        rows_ = [np.hstack([r, np.full((300, w - r.shape[1], 3), 30, np.uint8)]) if r.shape[1] < w else r for r in rows_]
        g = np.vstack([np.vstack([r, np.full((3, w, 3), 255, np.uint8)]) for r in rows_])
        cv2.imwrite(path, cv2.cvtColor(g, cv2.COLOR_RGB2BGR))
    if ant_examples:
        grid(ant_examples[-10:][::-1], f"{OUTD}/antenna_head_adds_most.png")
        grid(ant_examples[:10], f"{OUTD}/antenna_head_adds_least.png")
        print(f"\n  antenna examples -> {OUTD}/antenna_head_adds_{{most,least}}.png")
        print("    (green=GT antenna boundary, blue=already covered by our outline, red=ADDED by antenna head)")


if __name__ == "__main__":
    main()
