"""Aspect-bucket routing and letterboxing for the exterior segmentation BiRefNet.

A square 1024 input wastes ~76% of its pixels on background and letterbox, because car crops are
bimodal in aspect (side views ~2.5, straight-on ~1.2). Routing each crop to the nearest of five
landscape/portrait shapes -- all ~1.0 Mpx, all divisible by 32 -- was worth +0.013 BF@1 overall and
+0.047 on side views versus the square model, at equal compute.

Deploy contract: route on the aspect of the DETECTOR car box, letterbox within the chosen bucket.
"""
import math

import cv2
import numpy as np

# (W, H) per bucket. All /32 for TensorRT, all ~1.0 Mpx so compute is constant across buckets.
BUCKETS = {"P": (960, 1056), "A": (1088, 896), "B": (1248, 800), "C": (1376, 704), "D": (1600, 640)}
# Bucket centres in aspect (W/H). "P" is portrait; routing is nearest centre in LOG aspect.
CENTERS = {"P": 0.91, "A": 1.21, "B": 1.56, "C": 1.95, "D": 2.50}
PAD = 0.08          # fraction of the car box added on each side before letterboxing
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def route(aspect: float) -> str:
    """Bucket key for a car-box aspect ratio (width / height)."""
    return min(CENTERS, key=lambda k: abs(math.log(aspect / CENTERS[k])))


def crop_pad_box(box, H, W):
    """Expand a car box by PAD on each side, clipped to the frame."""
    x0, y0, x1, y1 = box
    px, py = int((x1 - x0) * PAD), int((y1 - y0) * PAD)
    return max(0, x0 - px), max(0, y0 - py), min(W, x1 + px), min(H, y1 + py)


def letterbox(arr, W, H, interp=cv2.INTER_LINEAR):
    """Resize preserving aspect, centred on a WxH canvas. Returns (canvas, (ox, oy, nw, nh))."""
    h, w = arr.shape[:2]
    s = min(W / w, H / h)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    r = cv2.resize(arr, (nw, nh), interpolation=interp)
    ox, oy = (W - nw) // 2, (H - nh) // 2
    if arr.ndim == 3:
        out = np.zeros((H, W, 3), arr.dtype)
    else:
        out = np.zeros((H, W), arr.dtype)
    out[oy:oy + nh, ox:ox + nw] = r
    return out, (ox, oy, nw, nh)


def normalize(img_rgb):
    """HWC uint8 RGB -> CHW float32, ImageNet-normalised."""
    return ((img_rgb / 255.0 - MEAN) / STD).transpose(2, 0, 1).astype(np.float32)
