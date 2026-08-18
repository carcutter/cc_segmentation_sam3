#!/usr/bin/env python3
"""
Does a 2-PASS holes cascade recover the holes the single car-crop pass loses?

The holes UNet was trained on tight window-region crops, so running it on the
wide car-box crop (what the detector gives at deploy) under-fires. Test whether:
  pass1 = holes coarse on the CAR-box crop (rough, wide framing)
  -> bbox of pass1 (+context) = window region
  pass2 = holes coarse on that window crop (native training framing)
  -> existing refiner
recovers the standalone-eval quality WITHOUT retraining.

Car crop here = GT union bbox (isolates the holes cascade from detector error).
Compares, per holes-positive test image, IoU vs GT blue:
  oracle   : crop around GT blue bbox            (upper bound, what standalone eval did)
  pass1    : single pass on car crop             (current broken end2end)
  pass2    : cascade, crop around pass1 bbox
  refined  : pass2 + holes refiner

Run from repo root:
  PYTHONPATH=. /home/rutger/miniconda3/envs/cc_sam3/bin/python carcutter/car_bbox_detector/probe_holes_cascade.py --n 40
"""
import argparse, csv
import numpy as np, cv2
from PIL import Image
import torch, segmentation_models_pytorch as smp
from carcutter.car_bbox_detector.gen_coarse_masks import coarse_prob
from carcutter.car_bbox_detector.refine_infer_eval import refine
from carcutter.car_bbox_detector.seg_eval import mask_iou
BLUE = (0, 0, 255)


def load_unet(ckpt, ch, enc, dev):
    m = smp.Unet(enc, encoder_weights=None, in_channels=ch, classes=1).to(dev).eval()
    st = torch.load(ckpt, map_location=dev); m.load_state_dict(st.get("model", st)); return m


def bboxmask_from(mask, pad_frac, H, W):
    """0/255 box mask around `mask`'s bbox, expanded by pad_frac on each side."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    ph, pw = int((y1-y0+1)*pad_frac), int((x1-x0+1)*pad_frac)
    y0, y1 = max(0, y0-ph), min(H-1, y1+ph); x0, x1 = max(0, x0-pw), min(W-1, x1+pw)
    m = np.zeros((H, W), np.uint8); m[y0:y1+1, x0:x1+1] = 255; return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="carcutter/car_bbox_detector/data/index.csv")
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    E = "carcutter/car_bbox_detector/experiments"
    holes = load_unet(f"{E}/unet_holes_v1/checkpoints/best.pt", 3, "efficientnet-b4", dev)
    holes_ref = load_unet(f"{E}/unet_refine_holes_v1/checkpoints/best.pt", 4, "efficientnet-b2", dev)
    rows = [r for r in csv.DictReader(open(args.index)) if r["split"] == "test"]

    agg = {k: [] for k in ("oracle", "pass1", "pass2", "refined")}
    fired1 = 0; n = 0
    for r in rows:
        rgb = np.array(Image.open(r["mask"]).convert("RGB"))
        blue = (rgb == BLUE).all(2); H, W = blue.shape
        if blue.sum() < 50:
            continue
        union = rgb.sum(2) > 0
        img = np.array(Image.open(r["image"]).convert("RGB"))
        n += 1

        # oracle: crop around GT blue bbox (what standalone eval used)
        oc = coarse_prob(holes, img, blue.astype(np.uint8)*255, 768, dev) > 0.5
        agg["oracle"].append(mask_iou(blue, oc))

        # pass1: car-box crop (GT union bbox as the car crop)
        carbox = bboxmask_from(union, 0.05, H, W)
        p1 = coarse_prob(holes, img, carbox, 768, dev) > 0.5
        agg["pass1"].append(mask_iou(blue, p1))

        # pass2: crop around pass1's bbox (+50% context), re-run native framing
        if p1.sum() >= 30:
            fired1 += 1
            box2 = bboxmask_from(p1, 0.5, H, W)
            p2prob = coarse_prob(holes, img, box2, 768, dev)
            p2 = p2prob > 0.5
            agg["pass2"].append(mask_iou(blue, p2))
            ref = refine(holes_ref, img, p2prob, dev) > 0.5
            agg["refined"].append(mask_iou(blue, ref))
        else:
            agg["pass2"].append(0.0); agg["refined"].append(0.0)

    m = lambda a: float(np.mean(a)) if a else float("nan")
    print(f"Holes cascade probe on {n} holes-positive test imgs "
          f"(pass1 fired enough for cascade on {fired1}/{n})\n")
    print(f"  {'stage':<10}{'mean IoU':>10}{'median':>10}")
    for k in ("oracle", "pass1", "pass2", "refined"):
        a = agg[k]
        print(f"  {k:<10}{m(a):>10.3f}{np.median(a):>10.3f}")


if __name__ == "__main__":
    main()
