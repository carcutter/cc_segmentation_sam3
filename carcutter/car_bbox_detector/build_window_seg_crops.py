#!/usr/bin/env python3
"""
Phase B segmenter crops: tight window-region crops (blue/tint clusters + pad) at 512,
target = blue see-through mask. Plus random non-window negatives per car. SZ=512 to match
both training and deploy (Phase-A lesson: keep input resolution consistent end-to-end).

Out: window_seg_crops/{train,val}/{images,labels}/<stem>__r<i>.png
Run from repo root:
  PYTHONPATH=. python carcutter/car_bbox_detector/build_window_seg_crops.py
"""
import argparse, csv, random
from pathlib import Path
from multiprocessing import Pool
import numpy as np, cv2
from PIL import Image
OUT = "carcutter/car_bbox_detector/window_seg_crops"
BLUE = (0, 0, 255)
MINPX = 150; DILATE = 17; PAD = 0.25; SZ = 512; N_NEG = 2


def work(args_t):
    r, split, seed = args_t
    rng = random.Random(seed)
    rgb = np.array(Image.open(r["mask"]).convert("RGB")); blue = (rgb == BLUE).all(2); union = rgb.sum(2) > 0
    if union.sum() < 1500:
        return 0
    img = np.array(Image.open(r["image"]).convert("RGB")); H, W = img.shape[:2]
    ys, xs = np.where(union); bx0, by0, bx1, by1 = xs.min(), ys.min(), xs.max(), ys.max()
    wrote = 0
    def save(x0, y0, x1, y1, tag):
        nonlocal wrote
        if x1 - x0 < 12 or y1 - y0 < 12: return
        im = cv2.resize(img[y0:y1, x0:x1], (SZ, SZ))
        lb = cv2.resize((blue[y0:y1, x0:x1] * 255).astype(np.uint8), (SZ, SZ), interpolation=cv2.INTER_NEAREST)
        stem = f"{Path(r['image']).stem}__{tag}"
        cv2.imwrite(f"{OUT}/{split}/images/{stem}.png", cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{OUT}/{split}/labels/{stem}.png", lb); wrote += 1
    if blue.sum() >= MINPX:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATE, DILATE))
        nr, _, rst, _ = cv2.connectedComponentsWithStats(cv2.dilate(blue.astype(np.uint8), k), 8)
        for i in range(1, nr):
            rx, ry, rw, rh, a = rst[i]
            if a < MINPX: continue
            px, py = int(rw * PAD), int(rh * PAD)
            save(max(0, rx - px), max(0, ry - py), min(W, rx + rw + px), min(H, ry + rh + py), f"r{i}")
    for j in range(N_NEG):
        cw = rng.randint((bx1 - bx0) // 6, max((bx1 - bx0) // 6 + 1, (bx1 - bx0) // 3)); ch = cw
        x0 = rng.randint(bx0, max(bx0, bx1 - cw)); y0 = rng.randint(by0, max(by0, by1 - ch))
        x1, y1 = min(W, x0 + cw), min(H, y0 + ch)
        if blue[y0:y1, x0:x1].sum() == 0:
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
        print(f"  {sp}: {n} crops ({npos} with windows)")


if __name__ == "__main__":
    main()
