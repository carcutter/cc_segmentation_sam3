#!/usr/bin/env python3
"""
Build the car-extent bbox dataset from the car_segmentation batches.

Target = bounding box of the UNION of all non-background pixels in the GT
multiclass mask (full car extent, incl. the white thin-structure/antenna class).
Single class: `car`.

Per image we record the union bbox, whether the white/antenna class is present
(`has_antenna`), and whether a production outline mask exists (`has_prod`).

Split (coverage-first, vs-prod comparison):
  - TEST  : stratified sample from the prod batches (by has_antenna), so model
            vs prod vs GT compare on the same images.
  - VAL   : sample from the remainder.
  - TRAIN : everything else (incl. the non-prod batches).

Outputs (in --out):
  index.csv                      one row per image (batch, paths, bbox, flags, split)
  coco_car_{train,val,test}.json COCO detection, 1 class `car`
  RUN.md                         provenance

Usage:
    python build_car_bbox_coco.py
"""

import argparse
import csv
import json
import os
import random
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = "/home/rutger/work/cc_segmentation_sam3/data/car_segmentation"
BATCHES = [
    "kw2551_car_segmentation", "kw2553_car_segmentation", "kw2602_car_segmentation",
    "kw2603_car_segmentation", "kw2605_car_segmentation", "kw2609_car_segmentation",
    "kw2612_car_segmentation", "kw2615_car_segmentation",
]
WHITE = (255, 255, 255)   # thin top structure / antenna class
CATEGORIES = [{"id": 1, "name": "car", "supercategory": "vehicle"}]


def raw_dir(batch_path: Path) -> Path | None:
    for name in ("raw", "raw_images"):
        if (batch_path / name).is_dir() and any((batch_path / name).iterdir()):
            return batch_path / name
    return None


def union_bbox(mask_rgb: np.ndarray):
    """xywh bbox of all non-black pixels; bool has_white."""
    nonbg = mask_rgb.sum(2) > 0
    ys, xs = np.where(nonbg)
    if len(ys) == 0:
        return None, False
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    has_white = bool((mask_rgb == WHITE).all(2).any())
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1], has_white


def _process(task: dict):
    """Worker: read one mask, return its row dict (or None). Top-level for Pool pickling."""
    mask = np.array(Image.open(task["mask"]).convert("RGB"))
    bbox, has_white = union_bbox(mask)
    if bbox is None:
        return None
    h, w = mask.shape[:2]
    return {
        "batch": task["batch"], "image": task["image"], "mask": task["mask"],
        "prod_outline": task["prod_outline"],
        "width": w, "height": h,
        "x": bbox[0], "y": bbox[1], "bw": bbox[2], "bh": bbox[3],
        "has_antenna": int(has_white),
        "has_prod": int(bool(task["prod_outline"])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", default="data")
    ap.add_argument("--test-frac", type=float, default=0.12, help="frac of prod-batch images for test")
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)
    root = Path(args.root)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # Build the task list (cheap path checks), then process masks in parallel.
    tasks = []
    for batch in BATCHES:
        bp = root / batch
        rd = raw_dir(bp)
        mdir = bp / "masks"
        if rd is None or not mdir.is_dir():
            print(f"  skip {batch}: raw={rd} masks={mdir.is_dir()}"); continue
        prod_dir = bp / "masks_prod" / "outline"
        for img_path in sorted(rd.iterdir()):
            mpath = mdir / f"{img_path.stem}.png"
            if not mpath.exists():
                continue
            prod_path = prod_dir / f"{img_path.stem}.png"
            tasks.append({
                "batch": batch, "image": str(img_path), "mask": str(mpath),
                "prod_outline": str(prod_path) if prod_path.exists() else "",
            })
    print(f"Reading {len(tasks)} masks across {Pool()._processes} workers ...")
    with Pool() as pool:
        rows = [r for r in tqdm(pool.imap_unordered(_process, tasks, chunksize=16),
                                total=len(tasks), unit="img") if r is not None]
    print(f"\nTotal images with bbox: {len(rows)}")
    print(f"  with antenna(white): {sum(r['has_antenna'] for r in rows)}")
    print(f"  with prod outline  : {sum(r['has_prod'] for r in rows)}")

    # ---- split: test stratified from prod images by has_antenna ----
    prod = [r for r in rows if r["has_prod"]]
    nonprod = [r for r in rows if not r["has_prod"]]
    test = []
    for flag in (0, 1):
        grp = [r for r in prod if r["has_antenna"] == flag]
        random.shuffle(grp)
        k = int(len(grp) * args.test_frac)
        for r in grp[:k]:
            r["split"] = "test"
        test += grp[:k]
    remaining = [r for r in rows if r.get("split") != "test"]
    random.shuffle(remaining)
    nval = int(len(remaining) * args.val_frac)
    for r in remaining[:nval]:
        r["split"] = "val"
    for r in remaining[nval:]:
        r["split"] = "train"

    counts = {s: sum(1 for r in rows if r["split"] == s) for s in ("train", "val", "test")}
    print(f"Split: {counts}")
    print(f"  test antenna cases: {sum(1 for r in rows if r['split']=='test' and r['has_antenna'])}")

    # ---- write index.csv ----
    with open(out / "index.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)

    # ---- write COCO per split ----
    def write_coco(split):
        imgs_c, anns_c, aid = [], [], 1
        for iid, r in enumerate((x for x in rows if x["split"] == split), 1):
            imgs_c.append({"id": iid, "file_name": r["image"], "width": r["width"],
                           "height": r["height"], "batch": r["batch"],
                           "has_antenna": r["has_antenna"]})
            anns_c.append({"id": aid, "image_id": iid, "category_id": 1,
                           "bbox": [r["x"], r["y"], r["bw"], r["bh"]],
                           "area": r["bw"] * r["bh"], "iscrowd": 0})
            aid += 1
        json.dump({"categories": CATEGORIES, "images": imgs_c, "annotations": anns_c},
                  open(out / f"coco_car_{split}.json", "w"))
        return len(imgs_c)

    n = {s: write_coco(s) for s in ("train", "val", "test")}
    print(f"COCO written: {n}")

    (out / "RUN.md").write_text(
        f"# Car-extent bbox dataset\n\n"
        f"Source: `{args.root}` (batches: {', '.join(BATCHES)})\n\n"
        f"Target: bbox of UNION of all non-bg pixels in GT masks (full car incl. antenna).\n"
        f"Single class `car`. seed={args.seed}, test_frac={args.test_frac}(prod, strat by antenna), val_frac={args.val_frac}.\n\n"
        f"Images: {len(rows)} | antenna: {sum(r['has_antenna'] for r in rows)} | prod: {sum(r['has_prod'] for r in rows)}\n"
        f"Split: {counts} | test antenna cases: {sum(1 for r in rows if r['split']=='test' and r['has_antenna'])}\n"
    )
    print(f"\nDone -> {out}/")


if __name__ == "__main__":
    main()
