#!/usr/bin/env python3
"""Fast loaders for the cached pipeline predictions (data/e2e_cache/<stem>.npz).
Recomputes assembled outline/punchout from the cached primitives on CPU (cheap)."""
import csv
from pathlib import Path
import numpy as np, cv2
from PIL import Image
from scipy.ndimage import binary_fill_holes
from carcutter.car_bbox_detector.antenna_fp_clean import keep_main

ROOT = "carcutter/car_bbox_detector"; CACHE = Path(f"{ROOT}/data/e2e_cache")
WHEEL_COLORS = [(255, 199, 200), (128, 128, 128), (200, 0, 255), (255, 165, 0)]


def _unpack(packed, H, W):
    if packed.size == 0: return np.zeros((H, W), bool)
    return np.unpackbits(packed)[:H * W].reshape(H, W).astype(bool)


def load_pred(stem):
    """returns dict with o, ant, hole2, outline, punchout (bool HxW), carbox; or None if no car box."""
    fp = CACHE / f"{stem}.npz"
    if not fp.exists(): return None
    d = np.load(fp); H, W = [int(x) for x in d["shape"]]
    if d["o"].size == 0: return None
    o = _unpack(d["o"], H, W); ant = _unpack(d["ant"], H, W); hole2 = _unpack(d["hole2"], H, W)
    outline = keep_main(o | ant, 9)
    punchout = (binary_fill_holes(outline) & ~outline) | hole2
    return {"o": o, "ant": ant, "hole2": hole2, "outline": outline, "punchout": punchout, "carbox": d["carbox"]}


def wheelmask(rgb, tol=20):
    m = np.zeros(rgb.shape[:2], bool); rint = rgb.astype(int)
    for c in WHEEL_COLORS:
        m |= np.abs(rint - np.array(c)).max(2) <= tol
    return m


def wheel_region(rgb):
    return binary_fill_holes(cv2.dilate(wheelmask(rgb).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)


def test_rows(index=f"{ROOT}/data/index.csv"):
    return [r for r in csv.DictReader(open(index)) if r["split"] == "test"]


def iter_cached(index=f"{ROOT}/data/index.csv"):
    """yield (row, rgb_gt, pred_dict) for every test image that has a cached prediction."""
    for r in test_rows(index):
        pred = load_pred(Path(r["image"]).stem)
        if pred is None: continue
        rgb = np.array(Image.open(r["mask"]).convert("RGB"))
        yield r, rgb, pred
