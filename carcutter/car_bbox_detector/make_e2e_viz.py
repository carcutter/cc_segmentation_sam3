#!/usr/bin/env python3
"""
Qualitative best/worst-case panels for the end-to-end production comparison.

Reads e2e_metrics.csv + cached per-image masks (masks/<stem>.npz: outline, holes,
antenna, carbox) and renders, per class, the images where OURS beats PROD most
(best) and least / loses most (worst). Each row:
    raw (+detector car box) | GT overlay | OURS overlay | PROD overlay
with the per-image metric in each title. Outline/holes ranked by ours-minus-prod
(BF@1 for outline, IoU for holes); antenna ranked by ours-vs-GT IoU (no prod class).

Run from repo root:
  PYTHONPATH=. /home/rutger/miniconda3/envs/cc_sam3/bin/python carcutter/car_bbox_detector/make_e2e_viz.py \
      --e2e carcutter/car_bbox_detector/data/e2e --k 6
"""
import argparse, csv, os
from pathlib import Path
from collections import defaultdict
import numpy as np, cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, WHITE = (0, 0, 255), (255, 255, 255)
GT_COLOR, OURS_COLOR, PROD_COLOR = (0, 220, 0), (0, 180, 255), (255, 60, 60)


def overlay(img, mask, color, alpha=0.45):
    out = img.copy()
    if mask is not None and mask.any():
        cmask = np.zeros_like(img); cmask[mask] = color
        out = cv2.addWeighted(out, 1.0, cmask, alpha, 0)
        # outline contour for crispness
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, color, 2)
    return out


def gt_masks(mask_path):
    rgb = np.array(Image.open(mask_path).convert("RGB"))
    return {"outline": rgb.sum(2) > 0, "holes": (rgb == BLUE).all(2), "antenna": (rgb == WHITE).all(2)}


def _binar(p):
    if not p or not os.path.exists(p):
        return None
    a = np.array(Image.open(p)); return a > 0 if a.ndim == 2 else a.sum(2) > 0


def prod_mask_for(cls, prod_outline):
    """Corrected prod references: outline = outline-punchout; holes = holes_tint;
    antenna = raw outline (to visualize prod clipping the antenna)."""
    if not prod_outline:
        return None
    if cls == "outline":
        ol = _binar(prod_outline); hp = _binar(prod_outline.replace("/outline/", "/holes_punchout/"))
        return (ol & ~hp) if (ol is not None and hp is not None) else ol
    if cls == "holes":
        return _binar(prod_outline.replace("/outline/", "/holes_tint/"))
    if cls == "antenna":
        return _binar(prod_outline)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2e", default="carcutter/car_bbox_detector/data/e2e")
    ap.add_argument("--index", default="carcutter/car_bbox_detector/data/index.csv")
    ap.add_argument("--k", type=int, default=6, help="best + worst count per class")
    args = ap.parse_args()
    e2e = Path(args.e2e); viz = e2e / "viz"; viz.mkdir(parents=True, exist_ok=True)
    idx = {r["image"]: r for r in csv.DictReader(open(args.index))}

    # metric per (image, cls, method) — use the CORRECTED rescore if present
    mfile = e2e / "e2e_metrics_corrected.csv"
    if not mfile.exists():
        mfile = e2e / "e2e_metrics.csv"
    M = defaultdict(dict)
    for r in csv.DictReader(open(mfile)):
        M[(r["image"], r["cls"])][r["method"]] = {k: float(r[k]) for k in ("iou", "bf1", "bf2", "bf3")}

    rank_metric = {"outline": "bf1", "holes": "iou", "antenna": "iou"}
    for cls in ("outline", "holes", "antenna"):
        rm = rank_metric[cls]
        scored = []
        for (img, c), meth in M.items():
            if c != cls or "ours" not in meth:
                continue
            ours = meth["ours"][rm]
            prod = meth["prod"][rm] if "prod" in meth else None
            delta = (ours - prod) if prod is not None else ours   # antenna: rank by absolute IoU
            scored.append((delta, ours, prod, img))
        if not scored:
            continue
        scored.sort()
        worst = scored[:args.k]
        best = scored[-args.k:][::-1]

        for tag, sel in (("best", best), ("worst", worst)):
            n = len(sel)
            fig, axes = plt.subplots(n, 4, figsize=(20, 5 * n))
            if n == 1:
                axes = axes[None, :]
            for row, (delta, ours_s, prod_s, imgp) in enumerate(sel):
                npz = e2e / "masks" / f"{Path(imgp).stem}.npz"
                if not npz.exists():
                    continue
                d = np.load(npz)
                img = np.array(Image.open(imgp).convert("RGB"))
                gts = gt_masks(idx[imgp]["mask"])
                ours_mask = d[cls].astype(bool)
                prod_mask = prod_mask_for(cls, idx[imgp].get("prod_outline", ""))
                # raw + detector car box
                raw = img.copy()
                if "carbox" in d:
                    x0, y0, x1, y1 = [int(v) for v in d["carbox"]]
                    cv2.rectangle(raw, (x0, y0), (x1, y1), (255, 255, 0), 3)
                panels = [
                    (raw, f"raw + det box"),
                    (overlay(img, gts[cls], GT_COLOR), "GT"),
                    (overlay(img, ours_mask, OURS_COLOR), f"OURS {rm}={ours_s:.3f}"),
                    (overlay(img, prod_mask, PROD_COLOR), f"PROD {rm}={prod_s:.3f}" if prod_s is not None else "PROD (n/a)"),
                ]
                for col, (pim, title) in enumerate(panels):
                    axes[row, col].imshow(pim); axes[row, col].set_title(title, fontsize=11)
                    axes[row, col].axis("off")
            fig.suptitle(f"{cls.upper()} — {tag} {n} (ranked by ours−prod {rm})", fontsize=15, y=0.998)
            fig.tight_layout()
            outp = viz / f"{cls}_{tag}.png"
            fig.savefig(outp, dpi=70, bbox_inches="tight"); plt.close(fig)
            print(f"  wrote {outp}")
    print(f"Done -> {viz}")


if __name__ == "__main__":
    main()
