#!/usr/bin/env python3
"""
Best/worst outline panels for ours-v2 (BiRefNet hybrid) vs prod, on the e2e test set.
Columns: raw+detector box | GT | OURS v2 (hybrid) | PROD (outline-punchout).
Ranked by per-image (hybrid BF@1 - prod BF@1). CPU-only (cached masks).

Run from repo root:
  PYTHONPATH=. python carcutter/car_bbox_detector/final_viz.py --k 6
"""
import argparse, csv, os
from pathlib import Path
import numpy as np, cv2
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from carcutter.car_bbox_detector.seg_eval import boundary_f
from carcutter.car_bbox_detector.antenna_fp_clean import keep_main

GT_C, OURS_C, PROD_C = (0, 220, 0), (0, 180, 255), (255, 60, 60)


def binar(p):
    if not p or not os.path.exists(p):
        return None
    a = np.array(Image.open(p)); return (a.sum(2) > 0) if a.ndim == 3 else (a > 0)


def overlay(img, mask, color, a=0.45):
    out = img.copy()
    if mask is not None and mask.any():
        c = np.zeros_like(img); c[mask] = color
        out = cv2.addWeighted(out, 1.0, c, a, 0)
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, color, 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2e", default="carcutter/car_bbox_detector/data/e2e")
    ap.add_argument("--index", default="carcutter/car_bbox_detector/data/index.csv")
    ap.add_argument("--k", type=int, default=6)
    args = ap.parse_args()
    idx = {Path(r["image"]).stem: r for r in csv.DictReader(open(args.index))}
    viz = Path(args.e2e) / "viz_v2"; viz.mkdir(parents=True, exist_ok=True)
    bf_out = Path(args.e2e) / "birefnet_outline"

    rows = []
    for npz in sorted((Path(args.e2e) / "masks").glob("*.npz")):
        r = idx.get(npz.stem)
        if r is None:
            continue
        bfp = bf_out / f"{npz.stem}.npy"
        po = r["prod_outline"]
        if not bfp.exists() or not (po and os.path.exists(po)):
            continue
        d = np.load(npz)
        gtu = np.array(Image.open(r["mask"]).convert("RGB")).sum(2) > 0
        if gtu.sum() < 20:
            continue
        hybrid = keep_main(np.load(bfp) | d["antenna"].astype(bool), 9)
        ol = binar(po); hp = binar(po.replace("/outline/", "/holes_punchout/"))
        prod = (ol & ~hp) if (ol is not None and hp is not None) else ol
        hb1 = boundary_f(gtu, hybrid, 1); pb1 = boundary_f(gtu, prod, 1)
        rows.append((hb1 - pb1, hb1, pb1, npz.stem, r, hybrid, prod, gtu))
    rows.sort()
    sel = {"worst": rows[:args.k], "best": rows[-args.k:][::-1]}
    for tag, items in sel.items():
        n = len(items)
        fig, ax = plt.subplots(n, 4, figsize=(20, 5 * n))
        if n == 1:
            ax = ax[None, :]
        for i, (dlt, hb1, pb1, stem, r, hybrid, prod, gtu) in enumerate(items):
            img = np.array(Image.open(r["image"]).convert("RGB"))
            raw = img.copy()
            d = np.load(Path(args.e2e) / "masks" / f"{stem}.npz")
            if "carbox" in d:
                x0, y0, x1, y1 = [int(v) for v in d["carbox"]]
                cv2.rectangle(raw, (x0, y0), (x1, y1), (255, 255, 0), 3)
            panels = [(raw, "raw + det box"), (overlay(img, gtu, GT_C), "GT"),
                      (overlay(img, hybrid, OURS_C), f"OURS v2  BF@1={hb1:.3f}"),
                      (overlay(img, prod, PROD_C), f"PROD  BF@1={pb1:.3f}")]
            for c, (pim, t) in enumerate(panels):
                ax[i, c].imshow(pim); ax[i, c].set_title(t, fontsize=11); ax[i, c].axis("off")
        fig.suptitle(f"OUTLINE ours-v2 (BiRefNet hybrid) vs prod — {tag} {n} (by BF@1 ours−prod)", fontsize=15, y=0.999)
        fig.tight_layout(); fig.savefig(viz / f"outline_v2_{tag}.png", dpi=70, bbox_inches="tight"); plt.close(fig)
        print(f"  wrote {viz}/outline_v2_{tag}.png")
    print(f"Done -> {viz}")


if __name__ == "__main__":
    main()
