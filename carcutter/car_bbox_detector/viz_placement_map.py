#!/usr/bin/env python3
"""
Visualize the antenna-placement-likelihood maps (per view) in the body frame.
Body bbox is drawn as a white rectangle (v=0 roofline at its top). Antennas live
just above the top edge; bright = likely antenna placement, dark = dead zone.

Run from repo root:
  PYTHONPATH=. python carcutter/car_bbox_detector/viz_placement_map.py
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="carcutter/car_bbox_detector/data/antenna_placement.npz")
    ap.add_argument("--out", default="carcutter/car_bbox_detector/data/antenna_placement.png")
    args = ap.parse_args()
    z = np.load(args.npz)
    U0, U1, V0, V1, GW, GH = z["extent"]
    views = [k[5:] for k in z.files if k.startswith("map__")]
    order = ["all", "side-left", "side-right", "front", "rear",
             "front-left", "front-right", "rear-left", "rear-right", "side"]
    views = [v for v in order if v in views] + [v for v in views if v not in order]
    n = len(views); cols = 5; rows = (n + cols - 1) // cols
    fig, ax = plt.subplots(rows, cols, figsize=(4 * cols, 4.2 * rows))
    ax = np.atleast_2d(ax)
    # body rectangle in (u,v) -> grid coords
    def to_grid(u, v):
        gi = (u - U0) / (U1 - U0) * GW
        gj = (v - V0) / (V1 - V0) * GH
        return gi, gj
    bx = [to_grid(0, 0)[0], to_grid(1, 0)[0]]
    by = [to_grid(0, 0)[1], to_grid(0, 1)[1]]
    for i, view in enumerate(views):
        a = ax[i // cols, i % cols]
        m = z[f"map__{view}"]
        a.imshow(m, cmap="inferno", origin="upper", aspect="auto")
        # draw body bbox
        a.add_patch(plt.Rectangle((bx[0], by[0]), bx[1] - bx[0], by[1] - by[0],
                                  fill=False, edgecolor="cyan", lw=1.5))
        a.axhline(by[0], color="cyan", ls=":", lw=0.8)  # roofline v=0
        cnt = int(z[f"cnt__{view}"]) if f"cnt__{view}" in z.files else 0
        hot = 100 * (m > 0.1).mean()
        a.set_title(f"{view}  (n={cnt}, hot={hot:.0f}%)", fontsize=10)
        a.set_xticks([]); a.set_yticks([])
    for j in range(n, rows * cols):
        ax[j // cols, j % cols].axis("off")
    fig.suptitle("Antenna-placement likelihood (body frame; cyan box=body bbox, dotted=roofline v=0)", fontsize=13)
    fig.tight_layout()
    fig.savefig(args.out, dpi=85, bbox_inches="tight")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
