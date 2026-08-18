#!/usr/bin/env python3
"""
Dedicated antenna UNet eval: white-pixel recall/IoU/boundary-F vs production outline.

GT = white (255,255,255) thin-structure pixels. Our pred = antenna UNet on the
tight crop around the white bbox (oracle crop = the detector's antenna box at
deploy), back-projected. Prod reference = how much of the white the production
OUTLINE mask covers (prod has no antenna head, so its outline's white-recall is
the bar — ~0.57 measured earlier; body-union UNet was ~0.55).

Run from repo root:
  PYTHONPATH=. /home/rutger/miniconda3/envs/cc_sam3/bin/python carcutter/car_bbox_detector/antenna_eval.py \
      --weights carcutter/car_bbox_detector/experiments/unet_antenna_v1/checkpoints/best.pt --n 290
"""
import argparse, csv, os
import numpy as np
from PIL import Image
import torch, segmentation_models_pytorch as smp
from carcutter.car_bbox_detector.seg_eval import unet_full_mask, boundary_f, mask_iou

WHITE = (0xff, 0xff, 0xff)


def recall(gt, pred):
    d = gt.sum(); return (gt & pred).sum()/d if d else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="carcutter/car_bbox_detector/experiments/unet_antenna_v1/checkpoints/best.pt")
    ap.add_argument("--index", default="carcutter/car_bbox_detector/data/index.csv")
    ap.add_argument("--crop-size", type=int, default=256)
    ap.add_argument("--n", type=int, default=290)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = smp.Unet("efficientnet-b4", encoder_weights=None, in_channels=3, classes=1).to(device).eval()
    st = torch.load(args.weights, map_location=device); model.load_state_dict(st.get("model", st))

    rows = [r for r in csv.DictReader(open(args.index)) if r["split"] == "test"][:args.n]
    U = {"iou": [], "rec": [], "bf1": [], "bf2": []}; P = {"rec": []}
    n = 0
    for r in rows:
        w = (np.array(Image.open(r["mask"]).convert("RGB")) == WHITE).all(2)
        if w.sum() < 20:
            continue
        n += 1
        pil = Image.open(r["image"]).convert("RGB")
        up = unet_full_mask(model, pil, w.astype(np.uint8) * 255, args.crop_size, device)  # 0/255 so crop works
        U["iou"].append(mask_iou(w, up)); U["rec"].append(recall(w, up))
        U["bf1"].append(boundary_f(w, up, 1)); U["bf2"].append(boundary_f(w, up, 2))
        if r["prod_outline"] and os.path.exists(r["prod_outline"]):
            pm = np.array(Image.open(r["prod_outline"])); pm = pm > 0 if pm.ndim == 2 else pm.sum(2) > 0
            P["rec"].append(recall(w, pm))
    m = lambda a: float(np.mean(a)) if a else float("nan")
    print(f"Antenna eval on {n} test imgs with white/antenna (>=20px)\n")
    print(f"ANTENNA-UNet : white IoU {m(U['iou']):.3f} | white-recall {m(U['rec']):.3f} | Boundary-F @1px {m(U['bf1']):.3f} @2px {m(U['bf2']):.3f}")
    print(f"PROD outline : white-recall {m(P['rec']):.3f}  (the bar; body-union UNet was ~0.55)")


if __name__ == "__main__":
    main()
