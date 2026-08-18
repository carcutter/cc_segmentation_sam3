#!/usr/bin/env python3
"""
Run the shipped segmentation pipeline over the quality-manager Exterior set (no GT available).

Pipeline (the June-13 final design, with the retrained unet_antenna_v3 zoom UNet):
  rfdetr_ca_unified_v1 (car class on the raw panel-1 photo) -> car box
  -> aspect-route to one of 5 buckets -> SPLICED bucketed BiRefNet (one encoder pass)
       ch0 outline   -> body silhouette
       ch1 tint      -> windows
       ch2 antenna   -> head mask (saved for reference, NOT unioned: the A/B showed the head is the arm
                        that produces stray sticks -- 4% of images vs the zoom UNet's 0.4%)
  same detector re-run on the car crop -> antenna boxes -> zoom UNet (EffNet-B4 @256) -> antenna mask
  outline  = keep_main(body | antenna, 9)
  antenna  = the pixels the zoom UNet contributes ON TOP of the bare body silhouette, kept as its own
             output so QA can judge/remove the antenna without disturbing the rest of the cutout
  punchout = binary_fill_holes(outline) & ~outline      (silhouette-structural; the hole-UNet refiner is
                                                         deliberately OFF per the June-12 ablation)

Writes per-image masks, an RGBA cutout, and a 5-up review sheet (input | ours | prod | QC-retouched).

Run: PYTHONPATH=.:carcutter/car_bbox_detector/birefnet/BiRefNet \
     python carcutter/car_bbox_detector/run_qm_eval.py --n 12
"""
import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np
import rfdetr
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from scipy.ndimage import binary_fill_holes

sys.path.insert(0, "carcutter/car_bbox_detector/birefnet/BiRefNet")
from carcutter.car_bbox_detector.antenna_fp_clean import keep_main
from carcutter.car_bbox_detector.build_spliced_eval import PROD5, TRI, Spliced, infer, load_bn
from carcutter.car_bbox_detector.cascade_eval import predict_union
from carcutter.car_bbox_detector.gen_coarse_masks import coarse_prob
from carcutter.car_bbox_detector.probe_birefnet_aspect import route

ROOT = "carcutter/car_bbox_detector"
EXP = f"{ROOT}/experiments"
CARDET = f"{EXP}/rfdetr_ca_unified_v1/checkpoint_best_ema.pth"
ANTUNET = f"{EXP}/unet_antenna_v3/checkpoints/best.pt"
SET = "/home/rutger/work/cc_segmentation_sam3/data/car_segmentation/qm_eval_202608_car_segmentation"
CAR_THR, ANT_THR = 0.3, 0.15
ANT_CLS = 1          # rfdetr_ca_unified: 0-indexed 0=car, 1=antenna
DET_THR = 0.5        # antenna proposal threshold on the car crop
ANT_SEG_THR = 0.5    # zoom-UNet mask threshold
ANT_SZ = 256
dev = "cuda"


def boxmask(box, H, W):
    m = np.zeros((H, W), np.uint8)
    x0, y0, x1, y1 = [int(v) for v in box]
    m[max(0, y0):min(H, y1 + 1), max(0, x0):min(W, x1 + 1)] = 255
    return m


def cutout_rgba(img, alpha):
    """Car on transparent background, alpha = outline minus see-through punchout."""
    out = np.zeros((*img.shape[:2], 4), np.uint8)
    out[..., :3] = img
    out[..., 3] = alpha.astype(np.uint8) * 255
    return out


def on_plate(img, alpha, shade=245):
    """Composite the cutout onto a flat light plate, for at-a-glance review."""
    plate = np.full_like(img, shade)
    a = alpha[..., None].astype(np.float32)
    return (img * a + plate * (1 - a)).astype(np.uint8)


def outline_overlay(img, outline, punchout):
    """Input photo with our contour drawn (green) and punchout holes marked (red)."""
    viz = img.copy()
    for mask, color in ((outline, (0, 255, 0)), (punchout, (255, 0, 0))):
        m = mask.astype(np.uint8)
        if m.sum() == 0:
            continue
        cnts, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(viz, cnts, -1, color, max(2, img.shape[1] // 500))
    return viz


def label(img, text, h=48):
    bar = np.full((h, img.shape[1], 3), 32, np.uint8)
    cv2.putText(bar, text, (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return np.concatenate([bar, img], 0)


def sheet(panels, width=1000):
    """Horizontal strip of equal-width labelled panels."""
    out = []
    for text, im in panels:
        s = width / im.shape[1]
        r = cv2.resize(im, (width, max(1, int(round(im.shape[0] * s)))))
        out.append(label(r, text))
    h = max(p.shape[0] for p in out)
    out = [np.pad(p, ((0, h - p.shape[0]), (0, 0), (0, 0)), constant_values=32) for p in out]
    return np.concatenate(out, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="limit images (0 = all)")
    ap.add_argument("--out", default=f"{SET}/ours")
    ap.add_argument("--ant-ckpt", default=ANTUNET)
    ap.add_argument("--no-antenna", action="store_true", help="body silhouette only, skip the zoom UNet")
    ap.add_argument("--only-missing", action="store_true", help="skip images that already have an outline mask")
    ap.add_argument("--no-fallback", action="store_true",
                    help="do not fall back to the full frame when the car detector finds nothing")
    args = ap.parse_args()

    for d in ("outline", "windows", "punchout", "antenna", "antenna_head", "cutout", "sheets"):
        os.makedirs(f"{args.out}/{d}", exist_ok=True)

    spliced = Spliced(load_bn(PROD5, 1, dev), load_bn(TRI, 3, dev)).eval()
    cardet = rfdetr.RFDETRMedium.from_checkpoint(CARDET)
    antunet = None
    if not args.no_antenna:
        antunet = smp.Unet("efficientnet-b4", encoder_weights=None, in_channels=3, classes=1).to(dev).eval()
        st = torch.load(args.ant_ckpt, map_location=dev)
        antunet.load_state_dict(st.get("model", st))
        print(f"antenna zoom UNet: {args.ant_ckpt}")

    srcs = sorted(glob.glob(f"{SET}/input/*.jpg"))
    if args.only_missing:
        srcs = [s for s in srcs
                if not os.path.exists(f"{args.out}/outline/{os.path.splitext(os.path.basename(s))[0]}.png")]
    if args.n:
        srcs = srcs[:args.n]
    nodet, fellback, t0 = [], [], time.time()

    for i, src in enumerate(srcs):
        stem = os.path.splitext(os.path.basename(src))[0]
        pil = Image.open(src).convert("RGB")
        img = np.array(pil)
        H, W = img.shape[:2]

        carbox = predict_union(cardet, pil, CAR_THR, ANT_THR)
        if carbox is not None:
            x0, y0, x1, y1 = [int(v) for v in carbox]
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        if carbox is None or x1 - x0 < 32 or y1 - y0 < 32:
            # The failures are close-framed vehicles that fill (and run off) the frame — exactly the
            # case the car-extent detector never trained on. The whole frame is then the right crop.
            if args.no_fallback:
                nodet.append(stem)
                continue
            fellback.append(stem)
            x0, y0, x1, y1 = 0, 0, W, H

        o, tint, ahead = infer(spliced, img, (x0, y0, x1, y1), route((x1 - x0) / (y1 - y0)), dev)

        # antenna proposals from the same detector, re-run on the car crop (recall needs the crop)
        ant = np.zeros((H, W), bool)
        if antunet is not None:
            dd = cardet.predict(Image.fromarray(img[y0:y1, x0:x1]), threshold=DET_THR)
            cls = np.asarray(dd.class_id) if getattr(dd, "class_id", None) is not None else np.zeros(len(dd.xyxy))
            for bi, b in enumerate(np.asarray(dd.xyxy)):
                if int(cls[bi]) != ANT_CLS:
                    continue
                bx0, by0, bx1, by1 = [int(v) for v in b]
                gbox = (x0 + bx0, y0 + by0, x0 + bx1, y0 + by1)
                ant |= coarse_prob(antunet, img, boxmask(gbox, H, W), ANT_SZ, dev) > ANT_SEG_THR

        body = keep_main(o, 9)
        outline = keep_main(o | ant, 9)
        # what the antenna model actually ADDED and that survived cleanup — the layer QA judges
        antenna_added = ant & ~body & outline
        punchout = binary_fill_holes(outline) & ~outline
        alpha = outline & ~punchout

        for name, m in (("outline", outline), ("windows", tint), ("punchout", punchout),
                        ("antenna", antenna_added), ("antenna_head", ahead)):
            Image.fromarray((m.astype(np.uint8) * 255)).save(f"{args.out}/{name}/{stem}.png", optimize=True)
        Image.fromarray(cutout_rgba(img, alpha)).save(f"{args.out}/cutout/{stem}.png")

        panels = [("1. input (panel 1)", img),
                  ("2. OURS - contour + punchout", outline_overlay(img, outline, punchout)),
                  ("3. OURS - cutout", on_plate(img, alpha))]
        for tag, d in (("4. current prod (panel 2)", "prod"), ("5. QC retouched (panel 3)", "qc")):
            p = f"{SET}/{d}/{stem}.jpg"
            if os.path.exists(p):
                panels.append((tag, np.array(Image.open(p).convert("RGB"))))
        cv2.imwrite(f"{args.out}/sheets/{stem}.jpg",
                    cv2.cvtColor(sheet(panels), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])

        if (i + 1) % 25 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(srcs)}  {el/(i+1):.2f}s/img  eta {el/(i+1)*(len(srcs)-i-1)/60:.1f}min", flush=True)

    print(f"\ndone: {len(srcs)-len(nodet)}/{len(srcs)} segmented in {(time.time()-t0)/60:.1f} min -> {args.out}")
    for label, lst in (("NO CAR DETECTED (skipped)", nodet), ("FULL-FRAME FALLBACK (no detection)", fellback)):
        if lst:
            print(f"{label} ({len(lst)}):")
            for s in lst:
                print("   ", s)


if __name__ == "__main__":
    main()
