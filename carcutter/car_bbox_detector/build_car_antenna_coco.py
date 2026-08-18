#!/usr/bin/env python3
"""
Build the 2-class dataset: `car` (full car-extent bbox, same target as v1) plus
`antenna` (the white thin-top-structure class, as separate connected-component
boxes). At inference the final box = union(car_box, predicted antenna boxes),
which is monotonic vs v1 — antenna boxes can only extend the top, never shrink it.

Reuses the SAME split as v1 (from data/index.csv) so eval is apples-to-apples.

Outputs: data/coco_ca_{train,val,test}.json  (categories: 1=car, 2=antenna)

Usage:
    python build_car_antenna_coco.py
"""
import argparse, csv, json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

WHITE = (255, 255, 255)
MIN_ANTENNA_AREA = 15      # drop white speckle smaller than this (px)
CATEGORIES = [{"id": 1, "name": "car", "supercategory": "vehicle"},
              {"id": 2, "name": "antenna", "supercategory": "vehicle"}]


def antenna_boxes(mask_path: str):
    """Connected-component xywh boxes of the white class."""
    m = np.array(Image.open(mask_path).convert("RGB"))
    white = (m == WHITE).all(2).astype(np.uint8)
    if white.sum() == 0:
        return []
    n, _, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= MIN_ANTENNA_AREA:
            out.append([int(x), int(y), int(w), int(h)])
    return out


def _process(row):
    boxes = antenna_boxes(row["mask"]) if int(row["has_antenna"]) else []
    return row["image"], boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/index.csv")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.index)))

    # antenna boxes only need re-reading masks for has_antenna rows
    ant_rows = [r for r in rows if int(r["has_antenna"])]
    print(f"Extracting antenna CC boxes from {len(ant_rows)} masks ...")
    with Pool() as pool:
        ant_map = dict(tqdm(pool.imap_unordered(_process, ant_rows, chunksize=16),
                            total=len(ant_rows), unit="img"))

    def write_split(split):
        imgs, anns, aid = [], [], 1
        for iid, r in enumerate((x for x in rows if x["split"] == split), 1):
            imgs.append({"id": iid, "file_name": r["image"], "width": int(r["width"]),
                         "height": int(r["height"]), "batch": r["batch"],
                         "has_antenna": int(r["has_antenna"])})
            # class 1: car = full extent box (same as v1)
            cx, cy, cw, ch = int(r["x"]), int(r["y"]), int(r["bw"]), int(r["bh"])
            anns.append({"id": aid, "image_id": iid, "category_id": 1,
                         "bbox": [cx, cy, cw, ch], "area": cw * ch, "iscrowd": 0}); aid += 1
            # class 2: antenna boxes (0+)
            for (ax, ay, aw, ah) in ant_map.get(r["image"], []):
                anns.append({"id": aid, "image_id": iid, "category_id": 2,
                             "bbox": [ax, ay, aw, ah], "area": aw * ah, "iscrowd": 0}); aid += 1
        json.dump({"categories": CATEGORIES, "images": imgs, "annotations": anns},
                  open(Path(args.out) / f"coco_ca_{split}.json", "w"))
        na = sum(1 for a in anns if a["category_id"] == 2)
        print(f"  {split}: {len(imgs)} imgs, {len(anns)} anns ({na} antenna boxes)")

    for s in ("train", "val", "test"):
        write_split(s)
    print("Done -> data/coco_ca_{train,val,test}.json")


if __name__ == "__main__":
    main()
