#!/usr/bin/env python3
"""
Holes/seethrough: our UNet vs production masks_prod/holes vs GT (blue), test split.

GT = blue (0,0,255) pixels. Our pred = holes UNet on the crop around the blue
region, back-projected to native res. Prod = masks_prod/holes (derived from the
prod_outline path). Reports IoU + Boundary-F@1/2/3 (holes are mid-size regions,
so IoU is meaningful here, plus boundary for compositing quality).

Run from repo root:
  PYTHONPATH=. /home/rutger/miniconda3/envs/cc_sam3/bin/python carcutter/car_bbox_detector/holes_eval.py \
      --weights carcutter/car_bbox_detector/experiments/unet_holes_v1/checkpoints/best.pt --n 300
"""
import argparse, csv
import numpy as np
from PIL import Image
import torch, segmentation_models_pytorch as smp
from carcutter.car_bbox_detector.seg_eval import unet_full_mask, boundary_f, mask_iou

BLUE = (0, 0, 255)


def prod_holes_path(prod_outline):
    return prod_outline.replace("/outline/", "/holes/") if prod_outline else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="carcutter/car_bbox_detector/experiments/unet_holes_v1/checkpoints/best.pt")
    ap.add_argument("--index", default="carcutter/car_bbox_detector/data/index.csv")
    ap.add_argument("--crop-size", type=int, default=768)
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = smp.Unet("efficientnet-b4", encoder_weights=None, in_channels=3, classes=1).to(device).eval()
    st = torch.load(args.weights, map_location=device); model.load_state_dict(st.get("model", st))

    rows = [r for r in csv.DictReader(open(args.index)) if r["split"] == "test"][:args.n]
    agg = {"unet": {}, "prod": {}}
    def add(d,k,v): d[k]=d.get(k,[])+[v]
    n_eval = 0
    import os
    for r in rows:
        gtb = (np.array(Image.open(r["mask"]).convert("RGB")) == BLUE).all(2)
        if gtb.sum() < 20:                       # only images that have see-through holes
            continue
        n_eval += 1
        pil = Image.open(r["image"]).convert("RGB")
        up = unet_full_mask(model, pil, gtb.astype(np.uint8) * 255, args.crop_size, device)  # 0/255 so crop-finder works
        preds = {"unet": up}
        php = prod_holes_path(r["prod_outline"])
        if php and os.path.exists(php):
            pm = np.array(Image.open(php)); preds["prod"] = (pm > 0 if pm.ndim == 2 else pm.sum(2) > 0)
        for nm, pr in preds.items():
            add(agg[nm], "iou", mask_iou(gtb, pr))
            for t in (1,2,3): add(agg[nm], f"bf{t}", boundary_f(gtb, pr, t))
    print(f"Holes eval on {n_eval} test imgs with see-through (GT blue)\n")
    for nm in ("unet","prod"):
        a = agg[nm]
        if not a: print(f"{nm}: (none)"); continue
        m = lambda k: np.mean(a[k])
        print(f"{nm.upper():5}: IoU {m('iou'):.4f} | Boundary-F @1px {m('bf1'):.3f} @2px {m('bf2'):.3f} @3px {m('bf3'):.3f}  (n={len(a['iou'])})")


if __name__ == "__main__":
    main()
