#!/usr/bin/env python3
"""
Build an index.csv-schema row set for the new kw2617 exterior batch so it can be merged for the
next/final training run. Excludes interior/close-up views; computes car-extent bbox from the GT
union mask; no masks_prod -> prod_outline empty, has_prod=0. Seeded train/val/test split.

Out: data/index_kw2617.csv  (same columns as data/index.csv)
Run: PYTHONPATH=. python carcutter/car_bbox_detector/build_kw2617_index.py
"""
import csv, glob, os, random
from multiprocessing import Pool
import numpy as np
from PIL import Image

BATCH = "kw2617_car_segmentation"
ROOT = f"/home/rutger/work/cc_segmentation_sam3/data/car_segmentation/{BATCH}"
OUT = "carcutter/car_bbox_detector/data/index_kw2617.csv"
WHITE = (255, 255, 255)
# interior / close-up tokens to exclude (keep side/front/rear/corner/straight/360/trunk)
BLOCK = ("detail", "interior", "engine", "wheel", "seat", "compartment",
         "driver", "codriver", "feature", "dashboard", "odometer", "badge", "logo")
COLS = ["batch", "image", "mask", "prod_outline", "width", "height",
        "x", "y", "bw", "bh", "has_antenna", "has_prod", "split"]


def work(img):
    b = os.path.basename(img).lower()
    if any(t in b for t in BLOCK):
        return ("blocked", None)
    mask = f"{ROOT}/masks/" + os.path.basename(img).replace(".jpg", ".png")
    if not os.path.exists(mask):
        return ("nomask", None)
    rgb = np.array(Image.open(mask).convert("RGB")); H, W = rgb.shape[:2]
    u = rgb.sum(2) > 0
    if u.sum() < 1500:
        return ("empty", None)
    ys, xs = np.where(u); x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    has_ant = int((rgb == WHITE).all(2).any())
    # seeded split by stem
    rng = random.Random(hash(os.path.basename(img)) & 0xffffffff)
    rv = rng.random(); split = "train" if rv < 0.85 else ("val" if rv < 0.93 else "test")
    return ("ok", {"batch": BATCH, "image": img, "mask": mask, "prod_outline": "",
                   "width": W, "height": H, "x": x0, "y": y0, "bw": x1 - x0 + 1, "bh": y1 - y0 + 1,
                   "has_antenna": has_ant, "has_prod": 0, "split": split})


def main():
    raws = sorted(glob.glob(f"{ROOT}/raw/*.jpg"))
    with Pool(16) as p:
        res = p.map(work, raws, chunksize=16)
    from collections import Counter
    stat = Counter(r[0] for r in res)
    rows = [r[1] for r in res if r[0] == "ok"]
    w = csv.DictWriter(open(OUT, "w", newline=""), fieldnames=COLS); w.writeheader(); w.writerows(rows)
    print(f"kept {len(rows)} exterior rows  | dropped: {dict(stat)}")
    print(f"splits: {Counter(r['split'] for r in rows)}")
    print(f"with antenna: {sum(r['has_antenna'] for r in rows)}  | -> {OUT}")


if __name__ == "__main__":
    main()
