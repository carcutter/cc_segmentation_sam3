#!/usr/bin/env python3
"""Visualize hole-region DETECTION boxes (RF-DETR) + hole SEGMENTER crops+masks."""
import json, random
from pathlib import Path
import numpy as np, cv2
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

COCO = "carcutter/car_bbox_detector/data/coco_hole_train.json"
SEG = "carcutter/car_bbox_detector/hole_seg_crops/train"
OUT = "carcutter/car_bbox_detector/data/holedata_viz.png"

coco = json.load(open(COCO))
byimg = {}
for a in coco["annotations"]:
    byimg.setdefault(a["image_id"], []).append(a["bbox"])
imgs = {im["id"]: im for im in coco["images"]}
ids_with = [i for i in byimg if len(byimg[i]) >= 1]
random.seed(3); random.shuffle(ids_with)
det_ids = ids_with[:4]

# crops with holes
seg_imgs = sorted(Path(SEG, "images").glob("*.png"))
pos_crops = [p for p in seg_imgs[:4000] if cv2.imread(str(Path(SEG, "labels", p.name)), 0).max() > 0]
random.shuffle(pos_crops); crop_sel = pos_crops[:8]

# --- detection boxes (4 imgs, large) ---
figd = plt.figure(figsize=(20, 6))
for k, iid in enumerate(det_ids):
    im = imgs[iid]; img = np.array(Image.open(im["file_name"]).convert("RGB"))
    for (x, y, w, h) in byimg[iid]:
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (255, 30, 30), 6)
    ax = figd.add_subplot(1, 4, k + 1); ax.imshow(img)
    ax.set_title(f"RF-DETR hole-region boxes (n={len(byimg[iid])})", fontsize=11); ax.axis("off")
figd.tight_layout(); figd.savefig(OUT.replace(".png", "_boxes.png"), dpi=100, bbox_inches="tight")
# --- seg crops + mask (8, large) ---
figs = plt.figure(figsize=(20, 10))
for k, p in enumerate(crop_sel):
    im = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
    lb = cv2.imread(str(Path(SEG, "labels", p.name)), 0) > 127
    ov = im.copy(); ov[lb] = (255, 0, 0); blend = (0.55 * im + 0.45 * ov).astype(np.uint8)
    ax = figs.add_subplot(2, 4, k + 1); ax.imshow(blend)
    ax.set_title("UNet crop (red=hole GT)", fontsize=11); ax.axis("off")
figs.tight_layout(); figs.savefig(OUT.replace(".png", "_crops.png"), dpi=100, bbox_inches="tight")
print(f"-> {OUT.replace('.png','_boxes.png')} and _crops.png")
