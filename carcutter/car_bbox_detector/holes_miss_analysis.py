#!/usr/bin/env python3
"""
Diagnose holes (tinted see-through window) misses: find images where OURS misses the
most GT-blue area, and render raw | GT blue | OURS | PROD(holes_tint). The key tell:
does PROD also miss it (-> label/appearance ceiling) or does prod get it (-> our weakness)?
Also reports, per case, ours-recall and prod-recall of the missed region.

Uses cached e2e_v2 holes (d['holes']). CPU.
Run from repo root:
  PYTHONPATH=. python carcutter/car_bbox_detector/holes_miss_analysis.py --k 8
"""
import argparse, csv, os
from pathlib import Path
import numpy as np, cv2
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
BLUE = (0, 0, 255)


def binar(p):
    if not p or not os.path.exists(p): return None
    a = np.array(Image.open(p)); return (a.sum(2) > 0) if a.ndim == 3 else (a > 0)


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
        rgb = np.array(Image.open(r["mask"]).convert("RGB")); blue = (rgb == BLUE).all(2)
        if blue.sum() < 200: continue
        ours = np.load(npz)["holes"].astype(bool)
        po = r["prod_outline"]
        prod = binar(po.replace("/outline/", "/holes_tint/")) if (po and os.path.exists(po)) else None
        miss = blue & ~ours
        rows.append(dict(stem=npz.stem, r=r, blue=blue, ours=ours, prod=prod,
                         missfrac=float(miss.sum()) / float(blue.sum()),
                         orec=float((blue & ours).sum()) / float(blue.sum()),
                         prec=(float((blue & prod).sum()) / float(blue.sum())) if prod is not None else None))
    rows.sort(key=lambda x: -x["missfrac"])
    # how often does PROD also miss what ours misses?
    both = [x for x in rows if x["prec"] is not None]
    print(f"holes-positive imgs: {len(rows)}. Of our biggest-miss cases, prod recall on same image:")
    sel = rows[:args.k]
    fig, ax = plt.subplots(len(sel), 4, figsize=(20, 5 * len(sel))); ax = np.atleast_2d(ax)
    for i, x in enumerate(sel):
        img = np.array(Image.open(x["r"]["image"]).convert("RGB"))
        def ov(m, col):
            o = img.copy()
            if m is not None and m.any(): o[m] = (0.45 * o[m] + 0.55 * np.array(col)).astype(np.uint8)
            return o
        pr = f"{x['prec']:.2f}" if x["prec"] is not None else "n/a"
        print(f"  {x['stem'][-28:]}  ours_recall={x['orec']:.2f}  prod_recall={pr}")
        panels = [(img, "raw"), (ov(x["blue"], [0, 80, 255]), "GT blue (window)"),
                  (ov(x["ours"], [0, 200, 255]), f"OURS recall={x['orec']:.2f}"),
                  (ov(x["prod"], [255, 80, 0]), f"PROD recall={pr}")]
        for c, (pim, t) in enumerate(panels):
            ax[i, c].imshow(pim); ax[i, c].set_title(t, fontsize=10); ax[i, c].axis("off")
    fig.suptitle("HOLES (tinted windows) — our biggest misses; does prod miss them too?", fontsize=14)
    fig.tight_layout(); out = Path(args.e2e) / "holes_misses.png"
    fig.savefig(out, dpi=78, bbox_inches="tight"); print(f"-> {out}")
    if both:
        print(f"\nmean prod_recall on our top-{len(sel)} miss cases: {np.mean([x['prec'] for x in sel if x['prec'] is not None]):.2f}")
        print(f"mean ours_recall on those: {np.mean([x['orec'] for x in sel]):.2f}")


if __name__ == "__main__":
    main()
