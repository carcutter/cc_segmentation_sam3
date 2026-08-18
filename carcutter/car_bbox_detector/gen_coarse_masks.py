#!/usr/bin/env python3
"""
Step 1 of the boundary-refinement prototype: cache the coarse 768-UNet masks.

Runs the trained car-seg UNet over every image (all splits) and saves its
full-resolution coarse probability mask (0-255 PNG). These become the
'coarse-mask' input channel for the native-res refiner, which learns to fix the
UNet's actual boundary errors (not a synthetic proxy).

Output: coarse_masks/{split}/{stem}.png   (0-255 = sigmoid prob)

Run from repo root:
  PYTHONPATH=. /home/rutger/miniconda3/envs/cc_sam3/bin/python carcutter/car_bbox_detector/gen_coarse_masks.py
"""
import argparse, csv
from pathlib import Path
import numpy as np, cv2, torch
from PIL import Image
import segmentation_models_pytorch as smp
from carcutter.unet.dataset import _crop_params_from_gt
from carcutter.car_bbox_detector.seg_eval import MEAN, STD


COLORS = {"white": (255, 255, 255), "blue": (0, 0, 255)}


def target_mask(rgb, target):
    return rgb.sum(2) > 0 if target == "union" else (rgb == COLORS[target]).all(2)


@torch.no_grad()
def coarse_prob(model, img, gt_mask, crop_size, device):
    # gt_mask must be 0/255 (the crop-finder thresholds >127)
    p = _crop_params_from_gt(gt_mask, 1.5); H, W = gt_mask.shape
    top, left, ch, cw = p if p else (0, 0, H, W)
    crop = img[top:top+ch, left:left+cw]
    side = max(ch, cw); xo, yo = (side-cw)//2, (side-ch)//2
    sq = np.zeros((side, side, 3), np.uint8); sq[yo:yo+ch, xo:xo+cw] = crop
    inp = cv2.resize(sq, (crop_size, crop_size))
    t = torch.from_numpy(((inp/255.0-MEAN)/STD).transpose(2,0,1)[None]).float().to(device)
    pr = torch.sigmoid(model(t)[0,0]).cpu().numpy()
    pr = cv2.resize(pr, (side, side))[yo:yo+ch, xo:xo+cw]
    full = np.zeros((H, W), np.float32); full[top:top+ch, left:left+cw] = pr
    return full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="carcutter/car_bbox_detector/experiments/unet_car_seg_v1/checkpoints/best.pt")
    ap.add_argument("--index", default="carcutter/car_bbox_detector/data/index.csv")
    ap.add_argument("--out", default="carcutter/car_bbox_detector/coarse_masks")
    ap.add_argument("--crop-size", type=int, default=768)
    ap.add_argument("--gt-color", default="union", choices=["union", "white", "blue"],
                    help="which class the coarse model targets (controls crop region)")
    ap.add_argument("--positives-only", action="store_true",
                    help="skip images with no target pixels (for white/blue)")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = smp.Unet(encoder_name="efficientnet-b4", encoder_weights=None, in_channels=3, classes=1).to(device).eval()
    st = torch.load(args.weights, map_location=device); model.load_state_dict(st.get("model", st))

    rows = list(csv.DictReader(open(args.index)))
    for s in ("train","val","test"): (Path(args.out)/s).mkdir(parents=True, exist_ok=True)
    done = 0
    for r in rows:
        out = Path(args.out)/r["split"]/f"{Path(r['image']).stem}.png"
        if out.exists():
            done += 1; continue
        gt = target_mask(np.array(Image.open(r["mask"]).convert("RGB")), args.gt_color)
        if args.positives_only and gt.sum() < 20:
            continue
        img = np.array(Image.open(r["image"]).convert("RGB"))
        pr = coarse_prob(model, img, gt.astype(np.uint8) * 255, args.crop_size, device)  # 0/255 for crop-finder
        Image.fromarray((pr*255).astype(np.uint8)).save(out)
        done += 1
        if done % 500 == 0: print(f"  {done}/{len(rows)}", flush=True)
    print(f"Done: {done} coarse masks -> {args.out}/")


if __name__ == "__main__":
    main()
