#!/usr/bin/env python3
"""
Phase B: CAR-CROP window-region detection set. Region box = bbox of a cluster of GT
tint/see-through windows (blue), so the detector proposes window regions on the car crop
for the dedicated see-through UNet. Saves crops + boxes in crop coords.

Outputs: data/window_det_crops/<stem>.jpg + data/coco_windowcrop_{train,val,test}.json
Run from repo root:
  PYTHONPATH=. python carcutter/car_bbox_detector/build_window_region_coco_crop.py
"""
import argparse, csv, json
from multiprocessing import Pool
from pathlib import Path
import cv2, numpy as np
from PIL import Image
from tqdm import tqdm
BLUE = (0, 0, 255)
MINPX = 150; DILATE = 17; BPAD = 0.12; CROP_PAD = 0.15
CROPDIR = "carcutter/car_bbox_detector/data/window_det_crops"
CATEGORIES = [{"id": 1, "name": "windowregion", "supercategory": "car"}]


def process(row):
    rgb = np.array(Image.open(row["mask"]).convert("RGB")); blue = (rgb == BLUE).all(2)
    img = np.array(Image.open(row["image"]).convert("RGB")); H, W = img.shape[:2]
    x, y, bw, bh = int(row["x"]), int(row["y"]), int(row["bw"]), int(row["bh"])
    px, py = int(bw * CROP_PAD), int(bh * CROP_PAD)
    cx0, cy0 = max(0, x - px), max(0, y - py); cx1, cy1 = min(W, x + bw + px), min(H, y + bh + py)
    crop = img[cy0:cy1, cx0:cx1]
    if crop.shape[0] < 32 or crop.shape[1] < 32:
        return row["image"], None
    boxes = []
    if blue.sum() >= MINPX:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATE, DILATE))
        nr, _, rst, _ = cv2.connectedComponentsWithStats(cv2.dilate(blue.astype(np.uint8), k), 8)
        for i in range(1, nr):
            rx, ry, rw, rh, a = rst[i]
            if a < MINPX: continue
            bpx, bpy = int(rw * BPAD), int(rh * BPAD)
            bx0 = max(cx0, rx - bpx) - cx0; by0 = max(cy0, ry - bpy) - cy0
            bx1 = min(cx1, rx + rw + bpx) - cx0; by1 = min(cy1, ry + rh + bpy) - cy0
            if bx1 - bx0 >= 8 and by1 - by0 >= 8:
                boxes.append([int(bx0), int(by0), int(bx1 - bx0), int(by1 - by0)])
    stem = Path(row["image"]).stem
    cv2.imwrite(f"{CROPDIR}/{stem}.jpg", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    return row["image"], {"file": f"{CROPDIR}/{stem}.jpg", "w": int(crop.shape[1]), "h": int(crop.shape[0]), "boxes": boxes}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--index", default="carcutter/car_bbox_detector/data/index.csv")
    ap.add_argument("--out", default="carcutter/car_bbox_detector/data"); args = ap.parse_args()
    Path(CROPDIR).mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(args.index)))
    with Pool(16) as p:
        info = dict(tqdm(p.imap_unordered(process, rows, chunksize=16), total=len(rows)))

    def write_split(split):
        imgs, anns, aid = [], [], 1
        for iid, r in enumerate((x for x in rows if x["split"] == split), 1):
            d = info.get(r["image"])
            if d is None: continue
            imgs.append({"id": iid, "file_name": d["file"], "width": d["w"], "height": d["h"], "batch": r["batch"]})
            for (x, y, w, h) in d["boxes"]:
                anns.append({"id": aid, "image_id": iid, "category_id": 1, "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0}); aid += 1
        json.dump({"categories": CATEGORIES, "images": imgs, "annotations": anns},
                  open(Path(args.out) / f"coco_windowcrop_{split}.json", "w"))
        print(f"  {split}: {len(imgs)} crops, {len(anns)} window-region boxes")
    for s in ("train", "val", "test"):
        write_split(s)
    print("Done -> coco_windowcrop_*.json + window_det_crops/")


if __name__ == "__main__":
    main()
