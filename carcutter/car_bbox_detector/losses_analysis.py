#!/usr/bin/env python3
"""
Is ours-v2 actually the better OUTLINE model, or do we win by pixels and lose by chunks?
Quantifies win vs loss MAGNITUDE and a "cut-off whole segment" metric, then renders our
worst losses so they can be eyeballed.

Per image (ours-v2 = e2e_v2 d['outline'] vs prod = outline-punchout vs GT union):
  - bf1/iou deltas (ours - prod)
  - missed_chunk = area of the LARGEST connected component of (GT & ~pred) / car area
    -> the biggest piece of car the model drops (a "cut-off segment"); compare ours vs prod
Renders the K worst losses (by prod_bf1 - ours_bf1): raw+box | GT | OURS (missed in red) |
PROD (missed in red), titled with bf1 + missed-chunk%.

Run from repo root:
  PYTHONPATH=. python carcutter/car_bbox_detector/losses_analysis.py --k 8
"""
import argparse, csv, os
from pathlib import Path
import numpy as np, cv2
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from carcutter.car_bbox_detector.seg_eval import boundary_f, mask_iou


def binar(p):
    if not p or not os.path.exists(p): return None
    a = np.array(Image.open(p)); return (a.sum(2) > 0) if a.ndim == 3 else (a > 0)


def max_missed_chunk(gt, pred):
    """Largest connected component of GT not covered by pred, as fraction of GT area."""
    miss = (gt & ~pred).astype(np.uint8)
    if miss.sum() == 0: return 0.0
    n, _, stats, _ = cv2.connectedComponentsWithStats(miss, 8)
    if n <= 1: return 0.0
    return float(stats[1:, cv2.CC_STAT_AREA].max()) / float(gt.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2e", default="carcutter/car_bbox_detector/data/e2e_v2")
    ap.add_argument("--index", default="carcutter/car_bbox_detector/data/index.csv")
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()
    idx = {Path(r["image"]).stem: r for r in csv.DictReader(open(args.index))}

    rows = []
    for npz in sorted((Path(args.e2e) / "masks").glob("*.npz")):
        r = idx.get(npz.stem)
        if r is None: continue
        po = r["prod_outline"]
        if not (po and os.path.exists(po)): continue
        d = np.load(npz)
        gt = np.array(Image.open(r["mask"]).convert("RGB")).sum(2) > 0
        if gt.sum() < 20: continue
        ours = d["outline"].astype(bool)
        ol = binar(po); hp = binar(po.replace("/outline/", "/holes_punchout/"))
        prod = (ol & ~hp) if (ol is not None and hp is not None) else ol
        rows.append(dict(stem=npz.stem, r=r,
                         ob1=boundary_f(gt, ours, 1), pb1=boundary_f(gt, prod, 1),
                         oiou=mask_iou(gt, ours), piou=mask_iou(gt, prod),
                         omiss=max_missed_chunk(gt, ours), pmiss=max_missed_chunk(gt, prod)))
    n = len(rows)
    db1 = np.array([x["ob1"] - x["pb1"] for x in rows])
    diou = np.array([x["oiou"] - x["piou"] for x in rows])
    wins = [x for x in rows if x["ob1"] > x["pb1"]]; losses = [x for x in rows if x["ob1"] < x["pb1"]]
    mn = lambda a: float(np.mean(a)) if len(a) else 0.0
    print(f"OUTLINE win/loss MAGNITUDE, n={n} (wins {len(wins)}, losses {len(losses)})\n")
    print(f"  mean BF@1 delta on WINS : {mn([x['ob1']-x['pb1'] for x in wins]):+.3f}")
    print(f"  mean BF@1 delta on LOSSES: {mn([x['ob1']-x['pb1'] for x in losses]):+.3f}")
    print(f"  mean IoU  delta on WINS : {mn([x['oiou']-x['piou'] for x in wins]):+.4f}")
    print(f"  mean IoU  delta on LOSSES: {mn([x['oiou']-x['piou'] for x in losses]):+.4f}")
    print(f"  worst single IoU loss   : {diou.min():+.4f}   worst single BF@1 loss: {db1.min():+.3f}")
    print(f"\n  'cut-off chunk' = largest dropped car component / car area:")
    print(f"    OURS : mean {mn([x['omiss'] for x in rows]):.4f}  p90 {np.quantile([x['omiss'] for x in rows],0.9):.4f}  max {max(x['omiss'] for x in rows):.4f}")
    print(f"    PROD : mean {mn([x['pmiss'] for x in rows]):.4f}  p90 {np.quantile([x['pmiss'] for x in rows],0.9):.4f}  max {max(x['pmiss'] for x in rows):.4f}")
    for thr in (0.02, 0.05, 0.10):
        print(f"    images dropping a chunk > {int(thr*100)}% of car:  OURS {sum(x['omiss']>thr for x in rows)}   PROD {sum(x['pmiss']>thr for x in rows)}")

    losses.sort(key=lambda x: x["ob1"] - x["pb1"])     # most negative first
    sel = losses[:args.k]
    fig, ax = plt.subplots(len(sel), 4, figsize=(20, 5 * len(sel))); ax = np.atleast_2d(ax)
    for i, x in enumerate(sel):
        r = x["r"]; d = np.load(Path(args.e2e) / "masks" / f"{x['stem']}.npz")
        img = np.array(Image.open(r["image"]).convert("RGB"))
        gt = np.array(Image.open(r["mask"]).convert("RGB")).sum(2) > 0
        ours = d["outline"].astype(bool)
        ol = binar(r["prod_outline"]); hp = binar(r["prod_outline"].replace("/outline/", "/holes_punchout/"))
        prod = (ol & ~hp) if (ol is not None and hp is not None) else ol
        raw = img.copy()
        if "carbox" in d:
            x0, y0, x1, y1 = [int(v) for v in d["carbox"]]; cv2.rectangle(raw, (x0, y0), (x1, y1), (255, 255, 0), 3)
        def ov(mask, missed):
            o = img.copy(); o[mask] = (0.5 * o[mask] + 0.5 * np.array([0, 180, 255])).astype(np.uint8)
            o[missed] = (255, 0, 0)        # GT car this mask DROPPED, in red
            return o
        panels = [(raw, "raw+box"), (img.copy(), "GT"),
                  (ov(ours, gt & ~ours), f"OURS bf1={x['ob1']:.2f} miss={100*x['omiss']:.1f}%"),
                  (ov(prod, gt & ~prod), f"PROD bf1={x['pb1']:.2f} miss={100*x['pmiss']:.1f}%")]
        # GT panel: green
        g = img.copy(); g[gt] = (0.5 * g[gt] + 0.5 * np.array([0, 220, 0])).astype(np.uint8); panels[1] = (g, "GT")
        for c, (pim, t) in enumerate(panels):
            ax[i, c].imshow(pim); ax[i, c].set_title(t, fontsize=10); ax[i, c].axis("off")
    fig.suptitle("OUTLINE — our WORST losses vs prod (red = car region the mask dropped)", fontsize=14)
    fig.tight_layout(); out = Path(args.e2e) / "outline_losses.png"
    fig.savefig(out, dpi=75, bbox_inches="tight"); print(f"\n-> {out}")


if __name__ == "__main__":
    main()
