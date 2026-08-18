#!/usr/bin/env python3
"""
Build a binary segmentation dataset for the UNet (carcutter/unet/train.py format):
<out>/{train,val,test}/{images,labels}.

--target selects what the binary label is, from the GT multiclass mask:
  union  : all non-bg pixels (car + antenna)      -> body silhouette (car-seg)
  white  : (255,255,255) thin top structure       -> antenna
  blue   : (0,0,255) see-through windows           -> holes/seethrough (#15)
--positives-only drops images whose target mask is empty (needed for white/blue,
which only exist on some images; detection of presence is the detector's job).

Reuses the SAME split as the detector (data/index.csv). image = symlink.

Usage:
    python build_seg_dataset.py --target union --out seg_data
    python build_seg_dataset.py --target blue  --positives-only --out holes_seg_data
    python build_seg_dataset.py --target white --positives-only --out antenna_seg_data
"""
import argparse, csv
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

COLORS = {"white": (255, 255, 255), "blue": (0, 0, 255)}


def target_mask(m, target):
    if target == "union":
        return m.sum(2) > 0
    return (m == COLORS[target]).all(2)


def _process(args):
    row, out, target, positives_only = args
    img, mask = Path(row["image"]), Path(row["mask"])
    m = np.array(Image.open(mask).convert("RGB"))
    binary = target_mask(m, target)
    if positives_only and binary.sum() < 20:
        return None
    split, stem = row["split"], img.stem
    Image.fromarray((binary.astype(np.uint8) * 255)).save(Path(out) / split / "labels" / f"{stem}.png")
    dst = Path(out) / split / "images" / f"{stem}{img.suffix}"
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(img.resolve())
    return split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/index.csv")
    ap.add_argument("--out", default="seg_data")
    ap.add_argument("--target", default="union", choices=["union", "white", "blue"])
    ap.add_argument("--positives-only", action="store_true")
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.index)))
    for s in ("train", "val", "test"):
        (Path(args.out) / s / "images").mkdir(parents=True, exist_ok=True)
        (Path(args.out) / s / "labels").mkdir(parents=True, exist_ok=True)
    print(f"target={args.target} positives_only={args.positives_only}: {len(rows)} imgs, {Pool()._processes} workers")
    tasks = [(r, args.out, args.target, args.positives_only) for r in rows]
    with Pool() as pool:
        splits = [s for s in tqdm(pool.imap_unordered(_process, tasks, chunksize=16),
                                  total=len(rows), unit="img") if s]
    print("per split:", dict(Counter(splits)))
    print(f"Done -> {args.out}/")


if __name__ == "__main__":
    main()
