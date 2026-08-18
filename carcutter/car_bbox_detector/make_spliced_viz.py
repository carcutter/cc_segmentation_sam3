#!/usr/bin/env python3
"""
Spliced single-pass model vs REAL production masks (masks_prod), per aspect, with win/lose examples
+ hole FP/FN audit. Aspects:
  outline  -> BF@1 vs prod outline; render wins/losses.
  holes    -> structural punchout (fill_holes(outline)&~outline) vs prod holes_punchout; INSTANCE
              FP/FN counts (ours & prod) + win/lose examples.
  windows  -> tint ch1 IoU vs prod holes_tint; wins/losses + interesting.
  antenna  -> ch2 vs GT (prod has NO antenna class); show quality wins/losses vs GT.
Out: experiments/spliced_viz/{outline,holes,windows,antenna}_{win,lose}.png + printed FP/FN table.

Run: PYTHONPATH=.:.../BiRefNet HF_HOME=... PY make_spliced_viz.py --n 930 --k 6
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
BLUE = (0, 0, 255); WHITE = (255, 255, 255); MINPX = 25
SIZES = [("small", 0, 0.005), ("med", 0.005, 0.02), ("large", 0.02, 1.0)]


def binar(p):
    if not p or not os.path.exists(p): return None
    a = np.array(Image.open(p)); return (a.sum(2) > 0) if a.ndim == 3 else (a > 0)


def comps(mask, ca, lo=0, hi=1.0):
    n, lbl, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return [(lbl == i, st[i, cv2.CC_STAT_AREA]) for i in range(1, n)
            if MINPX <= st[i, cv2.CC_STAT_AREA] and lo <= st[i, cv2.CC_STAT_AREA] / ca < hi]


def hole_stats(gt_holes, pred_holes, ca):
    """instance TP(recall)/FN over GT, and FP(spurious) over predicted, at 30% overlap."""
    gtc = comps(gt_holes, ca); prc = comps(pred_holes, ca)
    tp = sum(1 for g, a in gtc if (g & pred_holes).sum() >= 0.3 * a)
    fp = sum(1 for p, a in prc if (p & gt_holes).sum() < 0.3 * a)
    return len(gtc), tp, len(gtc) - tp, fp  # GT#, TP, FN, FP


def overlay(img, box, gt, pred, label, sz=360):
    x0, y0, x1, y1 = [int(v) for v in box]
    c = img[max(0,y0):y1, max(0,x0):x1].copy()
    g = gt[max(0,y0):y1, max(0,x0):x1]; p = pred[max(0,y0):y1, max(0,x0):x1] if pred is not None else None
    if p is not None:
        c[p] = (0.45 * c[p] + np.array([210, 30, 30])).clip(0, 255).astype(np.uint8)
    er = g ^ (cv2.erode(g.astype(np.uint8), np.ones((3,3), np.uint8)) > 0)
    c[er] = [0, 255, 0]
    c = cv2.resize(c, (sz, sz))
    cv2.putText(c, label, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    return c


def montage(items, path):  # items: list of (img, box, gt, ours, prod, lo, lp)
    rows = []
    for img, box, gt, ours, prod, lo, lp in items:
        tiles = [overlay(img, box, gt, ours, f"ours {lo}")]
        tiles.append(overlay(img, box, gt, prod, f"prod {lp}") if prod is not None else
                     np.full((360, 360, 3), 40, np.uint8))
        rows.append(np.hstack([np.hstack([t, np.full((360, 4, 3), 255, np.uint8)]) for t in tiles]))
    grid = np.vstack([np.vstack([r, np.full((4, r.shape[1], 3), 255, np.uint8)]) for r in rows])
    cv2.imwrite(path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


def render_row(model, r, aspect, dev):
    """Reload + re-infer one row, return the montage tuple for the given aspect."""
    img = np.array(Image.open(r["image"]).convert("RGB"))
    rgb = np.array(Image.open(r["mask"]).convert("RGB"))
    bw, bh = int(r["bw"]), int(r["bh"]); box = (int(r["x"]), int(r["y"]), int(r["x"])+bw, int(r["y"])+bh)
    ca = float((rgb.sum(2) > 0).sum()); po = r["prod_outline"]
    o, t, a = infer(model, img, box, route(bw/bh), dev)
    if aspect == "outline":
        gt = rgb.sum(2) > 0; ours = o; prod = binar(po)
        lo = f"{boundary_f(gt,ours,1):.2f}"; lp = f"{boundary_f(gt,prod,1):.2f}" if prod is not None else "n/a"
    elif aspect == "holes":
        u = rgb.sum(2) > 0; gt = binary_fill_holes(u) & ~u; ours = binary_fill_holes(o) & ~o
        prod = binar(po.replace("/outline/", "/holes_punchout/"))
        _, tp, fn, fp = hole_stats(gt, ours, ca); lo = f"TP{tp}/FN{fn}/FP{fp}"
        lp = ("TP{}/FN{}/FP{}".format(*hole_stats(gt, prod, ca)[1:])) if prod is not None else "n/a"
    elif aspect == "windows":
        gt = (rgb == BLUE).all(2); ours = t; prod = binar(po.replace("/outline/", "/holes_tint/"))
        lo = f"{mask_iou(gt,ours):.2f}"; lp = f"{mask_iou(gt,prod):.2f}" if prod is not None else "n/a"
    else:  # antenna (no prod class)
        gt = (rgb == WHITE).all(2); ours = a; prod = None
        lo = f"{mask_iou(gt,ours):.2f}"; lp = "n/a"
    return (img, box, gt, ours, prod, lo, lp)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=930); ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--index", default=f"{ROOT}/data/index.csv"); args = ap.parse_args(); dev = "cuda"
    os.makedirs(OUTD, exist_ok=True)
    model = Spliced(load_bn(PROD5, 1, dev), load_bn(TRI, 3, dev)).eval()
    rows = [r for r in csv.DictReader(open(args.index)) if r["split"] == "test"][:args.n]

    # PASS 1: scalars + refs only (no arrays kept) -> deltas + hole FP/FN totals
    rec = {a: [] for a in ("outline", "holes", "windows", "antenna")}  # (delta, r)
    H = {"ours": [0,0,0,0], "prod": [0,0,0,0]}
    for i, r in enumerate(rows):
        img = np.array(Image.open(r["image"]).convert("RGB"))
        rgb = np.array(Image.open(r["mask"]).convert("RGB"))
        gt_o, gt_t, gt_a = rgb.sum(2) > 0, (rgb == BLUE).all(2), (rgb == WHITE).all(2)
        if gt_o.sum() < 1500: continue
        ca = float(gt_o.sum()); bw, bh = int(r["bw"]), int(r["bh"])
        box = (int(r["x"]), int(r["y"]), int(r["x"])+bw, int(r["y"])+bh); po = r["prod_outline"]
        o, t, a = infer(model, img, box, route(bw/bh), dev)
        prod_o = binar(po); prod_h = binar(po.replace("/outline/","/holes_punchout/")) if po else None
        prod_t = binar(po.replace("/outline/","/holes_tint/")) if po else None
        if prod_o is not None:
            rec["outline"].append((boundary_f(gt_o,o,1) - boundary_f(gt_o,prod_o,1), r))
        gt_h = binary_fill_holes(gt_o) & ~gt_o; ours_h = binary_fill_holes(o) & ~o
        _, tp, fn, fp = hole_stats(gt_h, ours_h, ca)
        for k, v in zip(range(4), hole_stats(gt_h, ours_h, ca)): H["ours"][k] += v
        if prod_h is not None:
            g2, tp2, fn2, fp2 = hole_stats(gt_h, prod_h, ca)
            for k, v in zip(range(4), (g2,tp2,fn2,fp2)): H["prod"][k] += v
            rec["holes"].append(((tp-fp) - (tp2-fp2), r))
        if gt_t.sum() >= 60 and prod_t is not None:
            rec["windows"].append((mask_iou(gt_t,t) - mask_iou(gt_t,prod_t), r))
        if gt_a.sum() >= 20:
            rec["antenna"].append((mask_iou(gt_a,a), r))
        del img, rgb, o, t, a, prod_o, prod_h, prod_t
        if (i+1) % 150 == 0: print(f"  {i+1}/{len(rows)}", flush=True)

    print("\n=== PUNCHOUT-HOLE INSTANCE FP/FN (summed over test) ===")
    print(f"  {'':<6}{'GT#':>6}{'TP(hit)':>9}{'FN(miss)':>10}{'FP(spurious)':>14}{'recall':>8}{'prec':>7}")
    for k in ("ours", "prod"):
        g, tp, fn, fp = H[k]
        print(f"  {k:<6}{g:>6}{tp:>9}{fn:>10}{fp:>14}{100*tp/max(1,g):>7.0f}%{100*tp/max(1,tp+fp):>6.0f}%")

    # PASS 2: re-infer only the selected best/worst and render
    for a in ("outline", "holes", "windows"):
        rec[a].sort(key=lambda z: z[0])
        montage([render_row(model, r, a, dev) for _, r in rec[a][:args.k]], f"{OUTD}/{a}_lose.png")
        montage([render_row(model, r, a, dev) for _, r in rec[a][-args.k:][::-1]], f"{OUTD}/{a}_win.png")
    rec["antenna"].sort(key=lambda z: z[0])
    montage([render_row(model, r, "antenna", dev) for _, r in rec["antenna"][:args.k]], f"{OUTD}/antenna_worst.png")
    montage([render_row(model, r, "antenna", dev) for _, r in rec["antenna"][-args.k:][::-1]], f"{OUTD}/antenna_best.png")
    print(f"\n  montages -> {OUTD}/  (green=GT boundary, red=prediction; left=ours, right=prod)")


if __name__ == "__main__":
    main()
