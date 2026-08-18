#!/usr/bin/env python3
"""
Merged {car, antenna} detector dataset across BOTH framings, so ONE checkpoint serves the 2-pass design
(car on raw frame, antenna on car crop). Each index row -> up to 2 COCO images:
  RAW frame  : car=extent box (cat1) + antenna raw CC boxes (cat2)   [the only 'car' supervision]
  CROP tile  : antenna dilated-cluster boxes (cat2) in crop coords    [no car; crop fills frame]
                crop = car bbox + CROP_PAD (0.15), the deploy antenna framing.
Out: data/coco_mca_{train,val,test}.json  (cats 1=car, 2=antenna). Then build_rfdetr_dataset --prefix coco_mca.
Run from repo root: PYTHONPATH=. python carcutter/car_bbox_detector/build_merged_ca_coco.py
"""
import csv, json, os
from multiprocessing import Pool
from pathlib import Path
import cv2, numpy as np
from PIL import Image

ROOT = "carcutter/car_bbox_detector"; WHITE = (255, 255, 255)
A_RAW_MINPX = 15                       # raw antenna CC boxes (build_car_antenna_coco)
A_DILATE = 9; A_MINPX = 20; A_BPAD = 0.15; CROP_PAD = 0.15   # crop antenna clusters (build_multiclass_coco_crop)
CROPDIR = f"{ROOT}/data/hole_det_crops"
CATEGORIES = [{"id": 1, "name": "car", "supercategory": "vehicle"},
              {"id": 2, "name": "antenna", "supercategory": "vehicle"}]


def raw_antenna_boxes(white):
    n, _, st, _ = cv2.connectedComponentsWithStats(white.astype(np.uint8), 8)
    return [[int(st[i,0]), int(st[i,1]), int(st[i,2]), int(st[i,3])] for i in range(1, n) if st[i,4] >= A_RAW_MINPX]


def crop_antenna_clusters(white, cb):
    cx0, cy0, cx1, cy1 = cb; out = []
    if white.sum() < A_MINPX: return out
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (A_DILATE, A_DILATE))
    nr, _, rst, _ = cv2.connectedComponentsWithStats(cv2.dilate(white.astype(np.uint8), k), 8)
    for i in range(1, nr):
        rx, ry, rw, rh, a = rst[i]
        if a < A_MINPX: continue
        bpx, bpy = int(rw*A_BPAD), int(rh*A_BPAD)
        bx0 = max(cx0, rx-bpx)-cx0; by0 = max(cy0, ry-bpy)-cy0
        bx1 = min(cx1, rx+rw+bpx)-cx0; by1 = min(cy1, ry+rh+bpy)-cy0
        if bx1-bx0 >= 6 and by1-by0 >= 6: out.append([int(bx0), int(by0), int(bx1-bx0), int(by1-by0)])
    return out


def process(row):
    rgb = np.array(Image.open(row["mask"]).convert("RGB")); H, W = rgb.shape[:2]
    union = rgb.sum(2) > 0; white = (rgb == WHITE).all(2)
    x, y, bw, bh = int(row["x"]), int(row["y"]), int(row["bw"]), int(row["bh"])
    raw = {"file": row["image"], "w": W, "h": H,
           "anns": [(1, [x, y, bw, bh])] + [(2, b) for b in raw_antenna_boxes(white)]}
    crop = None
    if union.sum() >= 1500:
        px, py = int(bw*CROP_PAD), int(bh*CROP_PAD)
        cx0, cy0 = max(0, x-px), max(0, y-py); cx1, cy1 = min(W, x+bw+px), min(H, y+bh+py)
        cw, ch = cx1-cx0, cy1-cy0
        if cw >= 32 and ch >= 32:
            stem = Path(row["image"]).stem; cf = f"{CROPDIR}/{stem}.jpg"
            if not os.path.exists(cf):
                img = np.array(Image.open(row["image"]).convert("RGB"))
                cv2.imwrite(cf, cv2.cvtColor(img[cy0:cy1, cx0:cx1], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
            crop = {"file": cf, "w": cw, "h": ch,
                    "anns": [(2, b) for b in crop_antenna_clusters(white, (cx0, cy0, cx1, cy1))]}
    return row["image"], row["split"], raw, crop


def main():
    Path(CROPDIR).mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(f"{ROOT}/data/index.csv")))
    with Pool(16) as p:
        res = p.map(process, rows, chunksize=16)
    by_split = {"train": [], "val": [], "test": []}
    for _img, split, raw, crop in res:
        by_split[split].append((raw, crop))

    for split in ("train", "val", "test"):
        imgs, anns, iid, aid = [], [], 1, 1
        nraw = ncrop = nant = 0
        for raw, crop in by_split[split]:
            for entry, is_raw in ((raw, True), (crop, False)):
                if entry is None: continue
                imgs.append({"id": iid, "file_name": entry["file"], "width": entry["w"], "height": entry["h"]})
                for cat, b in entry["anns"]:
                    anns.append({"id": aid, "image_id": iid, "category_id": cat, "bbox": b,
                                 "area": b[2]*b[3], "iscrowd": 0}); aid += 1
                    if cat == 2: nant += 1
                iid += 1; nraw += is_raw; ncrop += (not is_raw)
        json.dump({"categories": CATEGORIES, "images": imgs, "annotations": anns},
                  open(f"{ROOT}/data/coco_mca_{split}.json", "w"))
        print(f"  {split}: {len(imgs)} imgs ({nraw} raw + {ncrop} crop), {len(anns)} anns ({nant} antenna)")
    print("Done -> data/coco_mca_{train,val,test}.json")


if __name__ == "__main__":
    main()
