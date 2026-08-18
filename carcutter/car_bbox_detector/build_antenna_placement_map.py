#!/usr/bin/env python3
"""
Antenna-placement-likelihood map (presence-INDEPENDENT spatial prior).

Idea (user): manufacturers place antennas in only a few spots on a car (rear-roof/
shark-fin, front-roof, A-pillar base, occasionally front fender) and ~never mid-door,
on glass, or off to the side. A stick/branch that the appearance UNet flags looks like
an antenna up close, but if it sits in a place an antenna is never placed, it's a FP.

We build, PER VIEW, a 2D heatmap of where GT antennas land in the *body* reference
frame (normalized by the body bbox = union minus antenna, so v≈0 is the roofline and
antennas protrude to v<0). The heatmap is the placement prior: hotspots where antennas
go, ~0 where they never do. At deploy it's projected onto any crop via (body bbox, view)
and used to prune antenna predictions that fall in low-likelihood zones — independent of
whether an antenna is actually present.

Output: data/antenna_placement.npz  (per-view HxW maps in [0,1] + grid extent).

Run from repo root:
  PYTHONPATH=. python carcutter/car_bbox_detector/build_antenna_placement_map.py
"""
import argparse, csv, re
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
WHITE = (255, 255, 255)
# normalized body-frame grid: u across body width, v down from body top (antennas at v<0)
U0, U1, V0, V1 = -0.20, 1.20, -0.70, 1.05
GW, GH = 80, 90          # grid width/height
SIGMA = 2.0


def view_of(p):
    m = re.search(r'-(front-left|front-right|rear-left|rear-right|side-left|side-right|front|rear|side)\.jpg$', p)
    return m.group(1) if m else "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="carcutter/car_bbox_detector/data/index.csv")
    ap.add_argument("--out", default="carcutter/car_bbox_detector/data/antenna_placement.npz")
    args = ap.parse_args()
    rows = [r for r in csv.DictReader(open(args.index)) if r["split"] in ("train", "val")]

    acc = {}; cnt = {}
    for r in rows:
        rgb = np.array(Image.open(r["mask"]).convert("RGB"))
        white = (rgb == WHITE).all(2)
        if white.sum() < 20:
            continue
        body = (rgb.sum(2) > 0) & (~white)            # car body without the antenna
        ys, xs = np.where(body)
        if len(xs) < 50:
            continue
        bx0, bx1, by0, by1 = xs.min(), xs.max(), ys.min(), ys.max()
        bw, bh = max(1, bx1 - bx0), max(1, by1 - by0)
        wy, wx = np.where(white)
        u = (wx - bx0) / bw; v = (wy - by0) / bh
        gi = ((u - U0) / (U1 - U0) * GW).astype(int)
        gj = ((v - V0) / (V1 - V0) * GH).astype(int)
        ok = (gi >= 0) & (gi < GW) & (gj >= 0) & (gj < GH)
        gi, gj = gi[ok], gj[ok]
        if len(gi) == 0:
            continue
        view = view_of(r["image"])
        h = np.zeros((GH, GW), np.float32)
        np.add.at(h, (gj, gi), 1.0)
        h /= h.sum()                                  # each car contributes equally
        acc.setdefault(view, np.zeros((GH, GW), np.float32))
        acc[view] += h; cnt[view] = cnt.get(view, 0) + 1

    maps = {}
    # 'all' = pooled across views (fallback when view unknown)
    pooled = np.zeros((GH, GW), np.float32)
    for view, h in acc.items():
        sm = gaussian_filter(h, SIGMA); sm /= (sm.max() + 1e-9)
        maps[view] = sm
        pooled += acc[view]
    maps["all"] = (lambda x: x / (x.max() + 1e-9))(gaussian_filter(pooled, SIGMA))
    np.savez_compressed(args.out, extent=np.array([U0, U1, V0, V1, GW, GH]),
                        **{f"map__{k}": v for k, v in maps.items()},
                        **{f"cnt__{k}": np.array(c) for k, c in cnt.items()})
    print(f"built placement maps for views: {sorted(cnt.items(), key=lambda x:-x[1])}")
    for k, v in maps.items():
        # quick describe: fraction of grid with prob>0.1 (smaller = more concentrated)
        print(f"  {k:<12} hotspot-area(prob>0.1)={100*(v>0.1).mean():.1f}%  peak@(v,u)="
              f"{np.unravel_index(v.argmax(), v.shape)}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
