#!/usr/bin/env python3
"""
Dedicated hole-SEGMENTER training crops. For each GT hole-region box (clustered small
holes), crop tight (padded) and upsample to 256 so the holes fill the frame; target =
small-hole mask in the crop. Plus a couple of RANDOM non-region crops per car as
negatives (door/hood/glass) so the segmenter learns "no hole here" -> precision.

Out: hole_seg_crops/{train,val}/{images,labels}/<stem>__r<i>.png
Run from repo root:
  PYTHONPATH=. python carcutter/car_bbox_detector/build_hole_seg_crops.py
"""
import argparse, csv, random
from pathlib import Path
from multiprocessing import Pool
import numpy as np, cv2
from PIL import Image
from scipy.ndimage import binary_fill_holes
OUT = "carcutter/car_bbox_detector/hole_seg_crops"
SMALL_FRAC = 0.006; MINPX = 25; DILATE = 35; PAD = 0.25; SZ = 256; N_NEG = 2


def small_and_regions(mask_path):
    rgb = np.array(Image.open(mask_path).convert("RGB")); union = rgb.sum(2) > 0
    ca = float(union.sum())
    if ca < 1500: return None
    holes = (binary_fill_holes(union) & ~union).astype(np.uint8)
    n, lbl, st, _ = cv2.connectedComponentsWithStats(holes, 8)
    small = np.zeros_like(holes)
    for i in range(1, n):
        if MINPX <= st[i, cv2.CC_STAT_AREA] < SMALL_FRAC * ca:
            small[lbl == i] = 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATE, DILATE))
    nr, _, rst, _ = cv2.connectedComponentsWithStats(cv2.dilate(small, k), 8)
    H, W = union.shape; regions = []
    for i in range(1, nr):
        x, y, w, h, _ = rst[i]
        px, py = int(w * PAD), int(h * PAD)
        regions.append((max(0, x - px), max(0, y - py), min(W, x + w + px), min(H, y + h + py)))
    return small, regions, union


def work(args_t):
    r, split, seed = args_t
    rng = random.Random(seed)
    res = small_and_regions(r["mask"])
    if res is None: return 0
    small, regions, union = res
    img = np.array(Image.open(r["image"]).convert("RGB")); H, W = img.shape[:2]
    ys, xs = np.where(union); bx0, by0, bx1, by1 = xs.min(), ys.min(), xs.max(), ys.max()
    wrote = 0
    def save(x0, y0, x1, y1, tag):
        nonlocal wrote
        if x1 - x0 < 12 or y1 - y0 < 12: return
        im = cv2.resize(img[y0:y1, x0:x1], (SZ, SZ))
        lb = cv2.resize((small[y0:y1, x0:x1] * 255).astype(np.uint8), (SZ, SZ), interpolation=cv2.INTER_NEAREST)
        stem = f"{Path(r['image']).stem}__{tag}"
        cv2.imwrite(f"{OUT}/{split}/images/{stem}.png", cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{OUT}/{split}/labels/{stem}.png", lb); wrote += 1
    for i, (x0, y0, x1, y1) in enumerate(regions):
        save(x0, y0, x1, y1, f"r{i}")
    # random negative crops inside the car bbox not overlapping any region
    for j in range(N_NEG):
        cw = rng.randint((bx1 - bx0) // 6, (bx1 - bx0) // 3); ch = cw
        x0 = rng.randint(bx0, max(bx0, bx1 - cw)); y0 = rng.randint(by0, max(by0, by1 - ch))
        x1, y1 = min(W, x0 + cw), min(H, y0 + ch)
        if small[y0:y1, x0:x1].sum() == 0:
            save(x0, y0, x1, y1, f"neg{j}")
    return wrote


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--index", default="carcutter/car_bbox_detector/data/index.csv")
    args = ap.parse_args()
    for sp in ("train", "val"):
        for d in ("images", "labels"): Path(f"{OUT}/{sp}/{d}").mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(args.index)))
    tasks = [(r, r["split"], i) for i, r in enumerate(rows) if r["split"] in ("train", "val")]
    with Pool(16) as p:
        res = p.map(work, tasks, chunksize=16)
    print(f"wrote {sum(res)} crops")
    for sp in ("train", "val"):
        n = len(list(Path(f"{OUT}/{sp}/images").glob("*.png")))
        npos = sum(1 for q in Path(f"{OUT}/{sp}/labels").glob("*.png") if cv2.imread(str(q), 0).max() > 0)
        print(f"  {sp}: {n} crops ({npos} with holes)")


if __name__ == "__main__":
    main()
