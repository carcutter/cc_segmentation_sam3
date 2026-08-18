#!/usr/bin/env python3
"""
Boundary-level segmentation eval (pixel-perfect standard): UNet vs production vs GT.

IoU on a car silhouette saturates near 1.0 and hides edge error — so we report
BOUNDARY metrics at native resolution:
  - mask IoU                     (reference only)
  - Boundary-F @ tol 1/2/3 px    (fraction of boundary pixels matched within tol)
  - Boundary-IoU @ d=2 px        (IoU restricted to a 2px band around boundaries)
  - antenna-region recall        (recall of GT antenna/white pixels — thin structure)

UNet runs on the GT-bbox crop (context 1.5, pad-square, --crop-size), prediction
back-projected to full native resolution, so the metric reflects real edge fidelity.

Run from repo root:
  PYTHONPATH=. /home/rutger/miniconda3/envs/cc_sam3/bin/python carcutter/car_bbox_detector/seg_eval.py \
      --weights carcutter/car_bbox_detector/experiments/unet_car_seg_v1/checkpoints/best.pt --n 200
"""
import argparse, csv
import numpy as np
import cv2
import torch
from PIL import Image
import segmentation_models_pytorch as smp
from carcutter.unet.dataset import _crop_params_from_gt

MEAN = np.array([0.485, 0.456, 0.406]); STD = np.array([0.229, 0.224, 0.225])


def boundary_pixels(mask, d):
    m = mask.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*d+1, 2*d+1))
    return (cv2.dilate(m, k) - cv2.erode(m, k)) > 0   # boundary band of width ~d


def boundary_f(gt, pred, tol):
    gb, pb = boundary_pixels(gt, 1), boundary_pixels(pred, 1)
    if gb.sum() == 0 and pb.sum() == 0:
        return 1.0
    if gb.sum() == 0 or pb.sum() == 0:
        return 0.0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*tol+1, 2*tol+1))
    gb_d = cv2.dilate(gb.astype(np.uint8), k) > 0
    pb_d = cv2.dilate(pb.astype(np.uint8), k) > 0
    prec = (pb & gb_d).sum() / pb.sum()      # pred boundary near a GT boundary
    rec  = (gb & pb_d).sum() / gb.sum()      # GT boundary near a pred boundary
    return 2*prec*rec/(prec+rec) if prec+rec > 0 else 0.0


def boundary_iou(gt, pred, d=2):
    gb, pb = boundary_pixels(gt, d), boundary_pixels(pred, d)
    gg, pp = gt.astype(bool) & gb, pred.astype(bool) & pb
    inter = (gg & pp).sum(); union = (gg | pp).sum()
    return inter/union if union > 0 else 1.0


def mask_iou(a, b):
    a, b = a.astype(bool), b.astype(bool)
    u = (a | b).sum(); return (a & b).sum()/u if u else 1.0


@torch.no_grad()
def unet_full_mask(model, pil, gt_mask, crop_size, device):
    """Crop around GT bbox (context 1.5), pad-square, resize, predict, back-project to full res."""
    params = _crop_params_from_gt(gt_mask, 1.5)
    H, W = gt_mask.shape
    if params is None:
        top, left, ch, cw = 0, 0, H, W
    else:
        top, left, ch, cw = params
    crop = np.array(pil)[top:top+ch, left:left+cw]
    side = max(ch, cw); xo, yo = (side-cw)//2, (side-ch)//2
    sq = np.zeros((side, side, 3), np.uint8); sq[yo:yo+ch, xo:xo+cw] = crop
    inp = cv2.resize(sq, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(((inp/255.0 - MEAN)/STD).transpose(2,0,1)[None]).float().to(device)
    logit = model(t)[0, 0]
    prob = torch.sigmoid(logit).cpu().numpy()
    prob_sq = cv2.resize(prob, (side, side), interpolation=cv2.INTER_LINEAR)
    crop_pred = prob_sq[yo:yo+ch, xo:xo+cw] > 0.5
    full = np.zeros((H, W), bool); full[top:top+ch, left:left+cw] = crop_pred
    return full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="carcutter/car_bbox_detector/experiments/unet_car_seg_v1/checkpoints/best.pt")
    ap.add_argument("--index", default="carcutter/car_bbox_detector/data/index.csv")
    ap.add_argument("--crop-size", type=int, default=768)
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = smp.Unet(encoder_name="efficientnet-b4", encoder_weights=None, in_channels=3, classes=1).to(device).eval()
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)

    rows = [r for r in csv.DictReader(open(args.index)) if r["split"] == "test"][:args.n]
    agg = {"unet": {}, "prod": {}}
    def add(d, k, v): d[k] = d.get(k, []) + [v]

    for r in rows:
        gt_rgb = np.array(Image.open(r["mask"]).convert("RGB"))
        gt = (gt_rgb.sum(2) > 0)
        white = (gt_rgb == (255,255,255)).all(2)        # antenna/thin pixels
        pil = Image.open(r["image"]).convert("RGB")
        up = unet_full_mask(model, pil, gt.astype(np.uint8), args.crop_size, device)
        preds = {"unet": up}
        if r["prod_outline"]:
            pm = np.array(Image.open(r["prod_outline"]))
            preds["prod"] = (pm > 0 if pm.ndim == 2 else pm.sum(2) > 0)
        for name, pr in preds.items():
            add(agg[name], "iou", mask_iou(gt, pr))
            for tol in (1,2,3): add(agg[name], f"bf{tol}", boundary_f(gt, pr, tol))
            add(agg[name], "biou", boundary_iou(gt, pr, 2))
            if white.sum() > 20:
                add(agg[name], "ant_recall", (pr & white).sum()/white.sum())

    print(f"Seg eval on {len(rows)} test imgs (UNet crop {args.crop_size}, back-projected to native)\n")
    for name in ("unet", "prod"):
        a = agg[name]
        if not a: continue
        m = lambda k: np.mean(a[k]) if a.get(k) else float('nan')
        print(f"{name.upper():5}: IoU {m('iou'):.4f} | Boundary-F @1px {m('bf1'):.3f} @2px {m('bf2'):.3f} @3px {m('bf3'):.3f} "
              f"| Boundary-IoU(d2) {m('biou'):.3f} | antenna-recall {m('ant_recall'):.3f} (n_ant={len(a.get('ant_recall',[]))})")


if __name__ == "__main__":
    main()
