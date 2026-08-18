#!/usr/bin/env python3
"""
1-class detection dataset for the dedicated HOLE-REGION detector. A region box = the
bbox of a CLUSTER of small structural holes (topological holes of the GT silhouette,
area < SMALL_FRAC of car), grouped by dilating the small-hole mask so a wheel's spoke
gaps / a roof rack's slots become one region. The detector learns the visual cue
(wheel/rack/bumper) so at deploy it proposes tight regions to a dedicated segmenter.

Outputs: data/coco_hole_{train,val,test}.json  (category 1=holeregion)
Run from repo root:
  PYTHONPATH=. python carcutter/car_bbox_detector/build_hole_region_coco.py
"""
import argparse, csv, json
from multiprocessing import Pool
from pathlib import Path
import cv2, numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes
from tqdm import tqdm
SMALL_FRAC = 0.006
MINPX = 25
DILATE = 35        # px to cluster nearby small holes into a region
PAD = 0.15         # pad region box
CATEGORIES = [{"id": 1, "name": "holeregion", "supercategory": "car"}]


def region_boxes(mask_path):
    rgb = np.array(Image.open(mask_path).convert("RGB")); union = rgb.sum(2) > 0
    ca = float(union.sum())
    if ca < 1500:
        return []
    holes = (binary_fill_holes(union) & ~union).astype(np.uint8)
    n, lbl, st, _ = cv2.connectedComponentsWithStats(holes, 8)
    small = np.zeros_like(holes)
    for i in range(1, n):
        if MINPX <= st[i, cv2.CC_STAT_AREA] < SMALL_FRAC * ca:
            small[lbl == i] = 1
    if small.sum() == 0:
        return []
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATE, DILATE))
    clustered = cv2.dilate(small, k)
    nr, _, rst, _ = cv2.connectedComponentsWithStats(clustered, 8)
    H, W = union.shape; out = []
    for i in range(1, nr):
        x, y, w, h, a = rst[i]
        px, py = int(w * PAD), int(h * PAD)
        x0, y0 = max(0, int(x - px)), max(0, int(y - py)); x1, y1 = min(W, int(x + w + px)), min(H, int(y + h + py))
        out.append([x0, y0, x1 - x0, y1 - y0])
    return out


def _process(row):
    return row["image"], region_boxes(row["mask"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="carcutter/car_bbox_detector/data/index.csv")
    ap.add_argument("--out", default="carcutter/car_bbox_detector/data")
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.index)))
    with Pool(16) as pool:
        bmap = dict(tqdm(pool.imap_unordered(_process, rows, chunksize=16), total=len(rows)))

    def write_split(split):
        imgs, anns, aid = [], [], 1
        for iid, r in enumerate((x for x in rows if x["split"] == split), 1):
            imgs.append({"id": iid, "file_name": r["image"], "width": int(r["width"]),
                         "height": int(r["height"]), "batch": r["batch"]})
            for (x, y, w, h) in bmap.get(r["image"], []):
                anns.append({"id": aid, "image_id": iid, "category_id": 1,
                             "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0}); aid += 1
        json.dump({"categories": CATEGORIES, "images": imgs, "annotations": anns},
                  open(Path(args.out) / f"coco_hole_{split}.json", "w"))
        print(f"  {split}: {len(imgs)} imgs, {len(anns)} hole-region boxes "
              f"({sum(1 for x in rows if x['split']==split and bmap.get(x['image']))} imgs w/ regions)")

    for s in ("train", "val", "test"):
        write_split(s)
    print("Done -> data/coco_hole_{train,val,test}.json")


if __name__ == "__main__":
    main()
