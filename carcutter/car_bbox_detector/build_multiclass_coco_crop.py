#!/usr/bin/env python3
"""
Phase C: UNIFIED multi-class car-crop detection set. One RF-DETR on the car crop proposes
candidates for all three dedicated UNets in a single pass:
  cat 1 antenna     (GT white clusters)        -> antenna UNet
  cat 2 holeregion  (GT topological small holes) -> hole-seg UNet
  cat 3 windowregion(GT tint/blue clusters)     -> window-seg UNet
Boxes are in CROP coords; reuses the existing identical car crops in data/hole_det_crops/
(same CROP_PAD=0.15 box as the per-class builders) — no new jpgs written.

Outputs: data/coco_multiclass_{train,val,test}.json
Run from repo root:
  PYTHONPATH=. python carcutter/car_bbox_detector/build_multiclass_coco_crop.py
"""
import argparse, csv, json, os
from multiprocessing import Pool
from pathlib import Path
import cv2, numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes
from tqdm import tqdm
WHITE = (255, 255, 255); BLUE = (0, 0, 255)
CROP_PAD = 0.15
# per-class cluster params (match the per-class builders)
H_SMALL_FRAC = 0.006; H_MINPX = 25; H_DILATE = 35; H_BPAD = 0.15
W_MINPX = 150; W_DILATE = 17; W_BPAD = 0.12
A_MINPX = 20; A_DILATE = 9; A_BPAD = 0.15
CROPDIR = "carcutter/car_bbox_detector/data/hole_det_crops"
CATEGORIES = [{"id": 1, "name": "antenna", "supercategory": "car"},
              {"id": 2, "name": "holeregion", "supercategory": "car"},
              {"id": 3, "name": "windowregion", "supercategory": "car"}]


def clusters(binmask, dilate, minpx, bpad, crop_box, W, H):
    """Return [x,y,w,h] boxes (crop coords) of dilated clusters of binmask, area>=minpx."""
    cx0, cy0, cx1, cy1 = crop_box
    out = []
    if binmask.sum() < minpx:
        return out
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
    nr, _, rst, _ = cv2.connectedComponentsWithStats(cv2.dilate(binmask.astype(np.uint8), k), 8)
    for i in range(1, nr):
        rx, ry, rw, rh, a = rst[i]
        if a < minpx: continue
        bpx, bpy = int(rw * bpad), int(rh * bpad)
        bx0 = max(cx0, rx - bpx) - cx0; by0 = max(cy0, ry - bpy) - cy0
        bx1 = min(cx1, rx + rw + bpx) - cx0; by1 = min(cy1, ry + rh + bpy) - cy0
        if bx1 - bx0 >= 6 and by1 - by0 >= 6:
            out.append([int(bx0), int(by0), int(bx1 - bx0), int(by1 - by0)])
    return out


def process(row):
    rgb = np.array(Image.open(row["mask"]).convert("RGB")); union = rgb.sum(2) > 0
    ca = float(union.sum())
    if ca < 1500:
        return row["image"], None
    H, W = rgb.shape[:2]
    x, y, bw, bh = int(row["x"]), int(row["y"]), int(row["bw"]), int(row["bh"])
    px, py = int(bw * CROP_PAD), int(bh * CROP_PAD)
    cx0, cy0 = max(0, x - px), max(0, y - py); cx1, cy1 = min(W, x + bw + px), min(H, y + bh + py)
    cw, ch = cx1 - cx0, cy1 - cy0
    if cw < 32 or ch < 32:
        return row["image"], None
    cb = (cx0, cy0, cx1, cy1)
    # class masks
    antenna = (rgb == WHITE).all(2)
    blue = (rgb == BLUE).all(2)
    holes = (binary_fill_holes(union) & ~union)
    small = np.zeros_like(holes, np.uint8)
    n, lbl, st, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8), 8)
    for i in range(1, n):
        if H_MINPX <= st[i, cv2.CC_STAT_AREA] < H_SMALL_FRAC * ca:
            small[lbl == i] = 1
    anns = []
    for b in clusters(antenna, A_DILATE, A_MINPX, A_BPAD, cb, W, H): anns.append((1, b))
    for b in clusters(small,   H_DILATE, H_MINPX, H_BPAD, cb, W, H): anns.append((2, b))
    for b in clusters(blue,    W_DILATE, W_MINPX, W_BPAD, cb, W, H): anns.append((3, b))
    stem = Path(row["image"]).stem
    cf = f"{CROPDIR}/{stem}.jpg"
    if not os.path.exists(cf):   # fallback: write crop if the reused dir is missing it
        img = np.array(Image.open(row["image"]).convert("RGB"))
        cv2.imwrite(cf, cv2.cvtColor(img[cy0:cy1, cx0:cx1], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    return row["image"], {"file": cf, "w": cw, "h": ch, "anns": anns}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--index", default="carcutter/car_bbox_detector/data/index.csv")
    ap.add_argument("--out", default="carcutter/car_bbox_detector/data"); args = ap.parse_args()
    Path(CROPDIR).mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(args.index)))
    with Pool(16) as p:
        info = dict(tqdm(p.imap_unordered(process, rows, chunksize=16), total=len(rows)))

    def write_split(split):
        imgs, annl, aid = [], [], 1
        per = {1: 0, 2: 0, 3: 0}
        for iid, r in enumerate((x for x in rows if x["split"] == split), 1):
            d = info.get(r["image"])
            if d is None: continue
            imgs.append({"id": iid, "file_name": d["file"], "width": d["w"], "height": d["h"], "batch": r["batch"]})
            for (cat, (bx, by, bw, bh)) in d["anns"]:
                annl.append({"id": aid, "image_id": iid, "category_id": cat, "bbox": [bx, by, bw, bh],
                             "area": bw * bh, "iscrowd": 0}); aid += 1; per[cat] += 1
        json.dump({"categories": CATEGORIES, "images": imgs, "annotations": annl},
                  open(Path(args.out) / f"coco_multiclass_{split}.json", "w"))
        print(f"  {split}: {len(imgs)} crops, {len(annl)} boxes  (antenna {per[1]}, hole {per[2]}, window {per[3]})")
    for s in ("train", "val", "test"):
        write_split(s)
    print("Done -> coco_multiclass_{train,val,test}.json")


if __name__ == "__main__":
    main()
