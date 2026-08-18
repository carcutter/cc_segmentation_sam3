#!/usr/bin/env python3
"""
Build training tiles for a DEDICATED small-hole UNet. Small holes cluster (validated) at
wheels (bottom corners), bumper (lower center), and roof rails (top band). We crop those
hole-prone REGIONS of the car bbox, upsample to high res (so small holes become large),
and the target = topological small holes (area < SMALL_FRAC of car) within the region.

Regions (fractions of car bbox): roof, lower-left, lower-right (overlap at bumper center).
Positives = tiles with small-hole pixels; a fraction of empty tiles kept as negatives so
the model learns to output nothing when this car has no holes there.

Out: smallhole_data/{train,val}/{images,labels}/<stem>__<region>.png
Run from repo root:
  PYTHONPATH=. python carcutter/car_bbox_detector/build_smallhole_tiles.py
"""
import argparse, csv, random
from pathlib import Path
from multiprocessing import Pool
import numpy as np, cv2
from PIL import Image
from scipy.ndimage import binary_fill_holes
OUT = "carcutter/car_bbox_detector/smallhole_data"
SMALL_FRAC = 0.006
MINPX = 25
TILE = 512
# region fractions of the car bbox: (x0,y0,x1,y1)
REGIONS = {"roof": (0.0, 0.0, 1.0, 0.30),
           "lowL": (0.0, 0.48, 0.62, 1.0),
           "lowR": (0.38, 0.48, 1.0, 1.0)}
import os
NEG_KEEP = float(os.environ.get("NEG_KEEP", 0.30))   # fraction of empty tiles to keep as negatives


def small_holes_mask(union, car_area):
    holes = (binary_fill_holes(union) & ~union).astype(np.uint8)
    n, lbl, st, _ = cv2.connectedComponentsWithStats(holes, 8)
    out = np.zeros_like(holes, bool)
    for i in range(1, n):
        a = st[i, cv2.CC_STAT_AREA]
        if MINPX <= a < SMALL_FRAC * car_area:
            out |= (lbl == i)
    return out


def work(args_t):
    r, split, seed = args_t
    rng = random.Random(seed)
    try:
        rgb = np.array(Image.open(r["mask"]).convert("RGB")); union = rgb.sum(2) > 0
        ca = float(union.sum())
        if ca < 1500: return 0
        img = np.array(Image.open(r["image"]).convert("RGB")); H, W = img.shape[:2]
        sh = small_holes_mask(union, ca)
        ys, xs = np.where(union); bx0, by0, bx1, by1 = xs.min(), ys.min(), xs.max(), ys.max()
        bw, bh = bx1 - bx0, by1 - by0
        wrote = 0
        for name, (fx0, fy0, fx1, fy1) in REGIONS.items():
            x0, y0 = int(bx0 + fx0 * bw), int(by0 + fy0 * bh)
            x1, y1 = int(bx0 + fx1 * bw), int(by0 + fy1 * bh)
            x0, y0 = max(0, x0), max(0, y0); x1, y1 = min(W, x1), min(H, y1)
            if x1 - x0 < 16 or y1 - y0 < 16: continue
            tile = cv2.resize(img[y0:y1, x0:x1], (TILE, TILE), interpolation=cv2.INTER_LINEAR)
            lab = cv2.resize((sh[y0:y1, x0:x1].astype(np.uint8) * 255), (TILE, TILE), interpolation=cv2.INTER_NEAREST)
            if lab.sum() == 0 and rng.random() > NEG_KEEP:
                continue
            stem = f"{Path(r['image']).stem}__{name}"
            cv2.imwrite(f"{OUT}/{split}/images/{stem}.png", cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))
            cv2.imwrite(f"{OUT}/{split}/labels/{stem}.png", lab)
            wrote += 1
        return wrote
    except Exception as e:
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="carcutter/car_bbox_detector/data/index.csv")
    args = ap.parse_args()
    for sp in ("train", "val"):
        for d in ("images", "labels"):
            Path(f"{OUT}/{sp}/{d}").mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(args.index)))
    tasks = [(r, r["split"], i) for i, r in enumerate(rows) if r["split"] in ("train", "val")]
    with Pool(16) as p:
        res = p.map(work, tasks, chunksize=16)
    print(f"wrote {sum(res)} tiles")
    for sp in ("train", "val"):
        n = len(list(Path(f"{OUT}/{sp}/images").glob("*.png")))
        npos = sum(1 for q in Path(f"{OUT}/{sp}/labels").glob("*.png")
                   if cv2.imread(str(q), 0).max() > 0)
        print(f"  {sp}: {n} tiles ({npos} with holes)")


if __name__ == "__main__":
    main()
