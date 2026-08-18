#!/usr/bin/env python3
"""
CORRECTED antenna fragmentation: prediction components vs GT components.

Naive '%pred has >1 component' is wrong — many cars genuinely have multiple
antennas/nubs (front+rear, whip+sharkfin), where >1 component is correct.
True discontinuity = a SINGLE GT antenna component predicted as >=2 disconnected
pieces. We measure per GT component.

Reports:
  - GT component distribution (how many images really have >1 antenna)
  - mean GT cc vs mean pred cc
  - of GT components the model actually hits, % that are predicted FRAGMENTED
    (broken into >1 piece) — the real floating-piece rate

Run from repo root:
  PYTHONPATH=. /home/rutger/miniconda3/envs/cc_sam3/bin/python carcutter/car_bbox_detector/antenna_frag_correct.py --n 290
"""
import argparse, csv
import numpy as np, cv2
from collections import Counter
from PIL import Image
import torch, segmentation_models_pytorch as smp
from carcutter.car_bbox_detector.seg_eval import unet_full_mask
WHITE = (255, 255, 255)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="carcutter/car_bbox_detector/experiments/unet_antenna_v1/checkpoints/best.pt")
    ap.add_argument("--index", default="carcutter/car_bbox_detector/data/index.csv")
    ap.add_argument("--crop-size", type=int, default=256); ap.add_argument("--n", type=int, default=290)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = smp.Unet("efficientnet-b4", encoder_weights=None, in_channels=3, classes=1).to(dev).eval()
    st = torch.load(args.weights, map_location=dev); model.load_state_dict(st.get("model", st))
    rows = [r for r in csv.DictReader(open(args.index)) if r["split"] == "test"][:args.n]

    gt_cc_dist = Counter(); gt_ccs, pred_ccs = [], []
    gt_comp_total = gt_comp_hit = gt_comp_fragmented = 0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    n = 0
    for r in rows:
        w = (np.array(Image.open(r["mask"]).convert("RGB")) == WHITE).all(2).astype(np.uint8)
        if w.sum() < 20: continue
        n += 1
        pred = unet_full_mask(model, Image.open(r["image"]).convert("RGB"), w*255, args.crop_size, dev).astype(np.uint8)
        n_gt, gt_lbl = cv2.connectedComponents(w)
        n_pr, _ = cv2.connectedComponents(pred)
        gt_cc_dist[n_gt-1] += 1; gt_ccs.append(n_gt-1); pred_ccs.append(n_pr-1)
        for i in range(1, n_gt):
            comp = (gt_lbl == i).astype(np.uint8)
            if comp.sum() < 15: continue
            gt_comp_total += 1
            pin = pred & (cv2.dilate(comp, k) > 0)         # pred overlapping this GT antenna (dilated)
            if pin.sum() < 0.15 * comp.sum(): continue     # essentially missed -> recall issue, not fragmentation
            gt_comp_hit += 1
            npieces, _ = cv2.connectedComponents(pin)
            if npieces-1 > 1: gt_comp_fragmented += 1
    print(f"Corrected antenna fragmentation on {n} test imgs\n")
    print(f"GT antenna components/image: {dict(sorted(gt_cc_dist.items()))}")
    print(f"  images with >1 GT antenna (real multi): {100*np.mean([c>1 for c in gt_ccs]):.1f}%")
    print(f"  mean GT cc {np.mean(gt_ccs):.2f} | mean PRED cc {np.mean(pred_ccs):.2f}")
    print(f"\nPer GT-antenna-component (the real test):")
    print(f"  GT components total {gt_comp_total} | hit by pred {gt_comp_hit}")
    print(f"  of hit components, FRAGMENTED into >1 piece: {gt_comp_fragmented} = {100*gt_comp_fragmented/max(1,gt_comp_hit):.1f}%")


if __name__ == "__main__":
    main()
