#!/usr/bin/env python3
"""
Antenna mask SHAPE / continuity analysis (+ plausible-fill fix).

Antennas are physically connected base->tip, but a thin-structure UNet tends to
predict FRAGMENTED masks (floating pieces where it missed faint mid-sections).
This quantifies the problem and tests a 'plausible fill': a vertical morphological
close that bridges vertical gaps between fragments.

Reports, on test images with antenna (white >=20px):
  - antenna-type split: nub (short, bbox height small) vs whip (tall)
  - fragmentation: mean #connected-components, % masks with >1 component (floating)
  - the vertical-close fill: #components + IoU/recall before vs after

Run from repo root:
  PYTHONPATH=. /home/rutger/miniconda3/envs/cc_sam3/bin/python carcutter/car_bbox_detector/antenna_shape_analysis.py \
      --weights carcutter/car_bbox_detector/experiments/unet_antenna_v1/checkpoints/best.pt --n 290
"""
import argparse, csv
import numpy as np, cv2
from PIL import Image
import torch, segmentation_models_pytorch as smp
from carcutter.car_bbox_detector.seg_eval import unet_full_mask, mask_iou

WHITE = (255, 255, 255)


def ncomp(m):
    n, _ = cv2.connectedComponents(m.astype(np.uint8)); return n - 1


def vfill(m, gap=41):
    """Bridge vertical gaps: morphological close with a tall thin kernel."""
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, gap))
    return cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, k) > 0


def recall(gt, p):
    d = gt.sum(); return (gt & p).sum()/d if d else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="carcutter/car_bbox_detector/experiments/unet_antenna_v1/checkpoints/best.pt")
    ap.add_argument("--index", default="carcutter/car_bbox_detector/data/index.csv")
    ap.add_argument("--crop-size", type=int, default=256)
    ap.add_argument("--gap", type=int, default=41, help="vertical-close kernel height (px) to bridge")
    ap.add_argument("--n", type=int, default=290)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = smp.Unet("efficientnet-b4", encoder_weights=None, in_channels=3, classes=1).to(device).eval()
    st = torch.load(args.weights, map_location=device); model.load_state_dict(st.get("model", st))

    rows = [r for r in csv.DictReader(open(args.index)) if r["split"] == "test"][:args.n]
    types = {"nub": 0, "whip": 0}
    nc_raw, nc_fix, frag_raw, frag_fix = [], [], [], []
    iou_raw, iou_fix, rec_raw, rec_fix = [], [], [], []
    n = 0
    for r in rows:
        w = (np.array(Image.open(r["mask"]).convert("RGB")) == WHITE).all(2)
        if w.sum() < 20:
            continue
        n += 1
        ys = np.where(w.any(1))[0]; gt_h = ys[-1]-ys[0]+1
        types["whip" if gt_h >= 120 else "nub"] += 1     # tall white bbox = long whip
        pil = Image.open(r["image"]).convert("RGB")
        pred = unet_full_mask(model, pil, w.astype(np.uint8)*255, args.crop_size, device)
        fixed = vfill(pred, args.gap)
        nc_raw.append(ncomp(pred)); nc_fix.append(ncomp(fixed))
        frag_raw.append(ncomp(pred) > 1); frag_fix.append(ncomp(fixed) > 1)
        iou_raw.append(mask_iou(w, pred)); iou_fix.append(mask_iou(w, fixed))
        rec_raw.append(recall(w, pred)); rec_fix.append(recall(w, fixed))
    m = lambda a: float(np.mean(a)) if a else float("nan")
    print(f"Antenna shape analysis on {n} test imgs (white>=20px)\n")
    print(f"Antenna-type split: nub {types['nub']} | whip(tall>=120px) {types['whip']}\n")
    print("                       raw-pred     after vertical-fill(gap={}px)".format(args.gap))
    print(f"  mean #components     {m(nc_raw):.2f}          {m(nc_fix):.2f}")
    print(f"  %fragmented (>1 cc)  {100*m(frag_raw):.1f}%         {100*m(frag_fix):.1f}%")
    print(f"  white IoU            {m(iou_raw):.3f}         {m(iou_fix):.3f}")
    print(f"  white recall         {m(rec_raw):.3f}         {m(rec_fix):.3f}")


if __name__ == "__main__":
    main()
