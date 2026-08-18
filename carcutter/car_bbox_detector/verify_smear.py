#!/usr/bin/env python3
"""
Verify the antenna placement-map "smear" inside the body box = REAL low antennas,
not annotation artifacts. Find cars whose GT antenna extends well below the roofline
(large fraction of white pixels at v>0.12 in body frame), and render them with the
antenna overlaid + roofline drawn, so we can eyeball what they actually are.

Run from repo root:
  PYTHONPATH=. python carcutter/car_bbox_detector/verify_smear.py --k 16
"""
import argparse, csv, re
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
WHITE = (255, 255, 255)


def view_of(p):
    m = re.search(r'-(front-left|front-right|rear-left|rear-right|side-left|side-right|front|rear|side)\.jpg$', p)
    return m.group(1) if m else "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="carcutter/car_bbox_detector/data/index.csv")
    ap.add_argument("--out", default="carcutter/car_bbox_detector/data/smear_verify.png")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--vthr", type=float, default=0.12, help="v below roofline to count as 'inside body'")
    args = ap.parse_args()
    rows = [r for r in csv.DictReader(open(args.index)) if r["split"] in ("train", "val")]

    cands = []; scanned = 0
    for r in rows:
        if len(cands) >= 250 or scanned >= 4000:
            break
        scanned += 1
        rgb = np.array(Image.open(r["mask"]).convert("RGB"))
        white = (rgb == WHITE).all(2)
        if white.sum() < 40:
            continue
        body = (rgb.sum(2) > 0) & (~white)
        ys, xs = np.where(body)
        if len(xs) < 50:
            continue
        by0, by1 = ys.min(), ys.max(); bh = max(1, by1 - by0)
        wy, wx = np.where(white)
        v = (wy - by0) / bh
        frac_in = float((v > args.vthr).mean())          # fraction of antenna pixels inside body
        maxv = float(v.max())
        if frac_in > 0.35:                               # antenna substantially below roofline
            cands.append((frac_in, maxv, view_of(r["image"]), r))
    cands.sort(key=lambda c: (c[0], c[1]), reverse=True)
    print(f"{len(cands)} cars have >35% of antenna pixels below v={args.vthr} (the 'smear')")
    by_view = {}
    for f, mv, vw, r in cands:
        by_view[vw] = by_view.get(vw, 0) + 1
    print("  by view:", sorted(by_view.items(), key=lambda x: -x[1]))

    sel = cands[:args.k]
    n = len(sel); cols = 4; rows_ = (n + cols - 1) // cols
    fig, ax = plt.subplots(rows_, cols, figsize=(5 * cols, 5 * rows_)); ax = np.atleast_2d(ax)
    for i, (frac_in, maxv, vw, r) in enumerate(sel):
        rgb = np.array(Image.open(r["mask"]).convert("RGB"))
        white = (rgb == WHITE).all(2); body = (rgb.sum(2) > 0) & (~white)
        img = np.array(Image.open(r["image"]).convert("RGB"))
        ys, xs = np.where(rgb.sum(2) > 0)
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        pad = int(0.08 * (x1 - x0))
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad); x1, y1 = min(img.shape[1], x1 + pad), min(img.shape[0], y1 + pad)
        crop = img[y0:y1, x0:x1].copy()
        wc = white[y0:y1, x0:x1]
        ov = crop.copy(); ov[wc] = (255, 30, 30)
        crop = (0.5 * crop + 0.5 * ov).astype(np.uint8)
        bys = np.where(body[y0:y1].any(1))[0]
        roof = bys.min() if len(bys) else 0            # body top within crop
        a = ax[i // cols, i % cols]
        a.imshow(crop); a.axhline(roof, color="cyan", ls=":", lw=1.5)
        a.set_title(f"{vw}  in-body={100*frac_in:.0f}%  maxv={maxv:.2f}", fontsize=10)
        a.axis("off")
    for j in range(n, rows_ * cols):
        ax[j // cols, j % cols].axis("off")
    fig.suptitle("Cars whose GT antenna (red) extends BELOW the roofline (cyan) — verifying the smear", fontsize=13)
    fig.tight_layout(); fig.savefig(args.out, dpi=80, bbox_inches="tight")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
