#!/usr/bin/env python3
"""
Step 3: stitched native-res refinement inference + boundary eval.

For each test image: take the coarse 768-UNet mask, tile the boundary band into
overlapping 256px NATIVE windows, run the 4ch refiner on each, average overlaps,
and replace the coarse mask inside the band. Reports COARSE vs REFINED vs PROD
on boundary metrics — the test of whether native-res refinement beats prod 0.708.

Run from repo root:
  PYTHONPATH=. /home/rutger/miniconda3/envs/cc_sam3/bin/python carcutter/car_bbox_detector/refine_infer_eval.py \
      --weights carcutter/car_bbox_detector/experiments/unet_refine_v1/checkpoints/best.pt --n 200
"""
import argparse, csv
from pathlib import Path
import numpy as np, cv2, torch
from PIL import Image
import segmentation_models_pytorch as smp
from carcutter.car_bbox_detector.seg_eval import boundary_f, boundary_iou, mask_iou

MEAN = np.array([0.485,0.456,0.406], np.float32); STD = np.array([0.229,0.224,0.225], np.float32)
ROOT = "/home/rutger/work/cc_segmentation_sam3_clean/carcutter/car_bbox_detector"


def band(m, d):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*d+1, 2*d+1))
    return (cv2.dilate(m, k) - cv2.erode(m, k)) > 0


@torch.no_grad()
def refine(model, img, coarse, device, patch=256, stride=128, band_px=48):
    H, W = coarse.shape
    cb = (coarse > 0.5).astype(np.uint8)
    region = band(cb, band_px)  # area to refine
    acc = np.zeros((H, W), np.float32); cnt = np.zeros((H, W), np.float32)
    tiles = []
    for top in range(0, max(1, H-patch)+1, stride):
        for left in range(0, max(1, W-patch)+1, stride):
            t = min(top, H-patch); l = min(left, W-patch); t = max(0, t); l = max(0, l)
            if region[t:t+patch, l:l+patch].any():
                tiles.append((t, l))
    for t, l in tiles:
        ip = img[t:t+patch, l:l+patch]; cm = coarse[t:t+patch, l:l+patch]
        x = np.concatenate([((ip/255.0-MEAN)/STD).transpose(2,0,1), cm[None]], 0)[None].astype(np.float32)
        pr = torch.sigmoid(model(torch.from_numpy(x).to(device))[0,0]).cpu().numpy()
        acc[t:t+patch, l:l+patch] += pr; cnt[t:t+patch, l:l+patch] += 1
    refined = coarse.copy()
    m = cnt > 0; refined[m] = acc[m]/cnt[m]
    return refined


def prod_mask(path):
    a = np.array(Image.open(path)); return a > 0 if a.ndim == 2 else a.sum(2) > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=f"{ROOT}/experiments/unet_refine_v1/checkpoints/best.pt")
    ap.add_argument("--index", default=f"{ROOT}/data/index.csv")
    ap.add_argument("--coarse", default=f"{ROOT}/coarse_masks")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--gt-color", default="union", choices=["union", "white", "blue"])
    ap.add_argument("--prod-kind", default="outline", choices=["outline", "holes"])
    ap.add_argument("--positives-only", action="store_true")
    args = ap.parse_args()
    COLORS = {"white": (255, 255, 255), "blue": (0, 0, 255)}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = smp.Unet("efficientnet-b2", encoder_weights=None, in_channels=4, classes=1).to(device).eval()
    st = torch.load(args.weights, map_location=device); model.load_state_dict(st.get("model", st))

    rows = [r for r in csv.DictReader(open(args.index)) if r["split"] == "test"][:args.n]
    agg = {"coarse": {}, "refined": {}, "prod": {}}
    def add(d,k,v): d[k]=d.get(k,[])+[v]
    for r in rows:
        cp = Path(args.coarse)/"test"/f"{Path(r['image']).stem}.png"
        if not cp.exists(): continue
        rgb = np.array(Image.open(r["mask"]).convert("RGB"))
        gt = (rgb.sum(2) > 0) if args.gt_color == "union" else (rgb == COLORS[args.gt_color]).all(2)
        if args.positives_only and gt.sum() < 20: continue
        img = np.array(Image.open(r["image"]).convert("RGB"))
        coarse = np.array(Image.open(cp)).astype(np.float32)/255.0
        refined = refine(model, img, coarse, device)
        masks = {"coarse": coarse > 0.5, "refined": refined > 0.5}
        prodp = r["prod_outline"].replace("/outline/", f"/{args.prod_kind}/") if r["prod_outline"] else ""
        import os
        if prodp and os.path.exists(prodp): masks["prod"] = prod_mask(prodp)
        for nm, pr in masks.items():
            add(agg[nm], "iou", mask_iou(gt, pr))
            for t in (1,2,3): add(agg[nm], f"bf{t}", boundary_f(gt, pr, t))
            add(agg[nm], "biou", boundary_iou(gt, pr, 2))
    print(f"Refiner stitched eval on {len(rows)} imgs\n")
    for nm in ("coarse","refined","prod"):
        a = agg[nm]
        if not a: continue
        m = lambda k: np.mean(a[k])
        print(f"{nm:8}: IoU {m('iou'):.4f} | Boundary-F @1px {m('bf1'):.3f} @2px {m('bf2'):.3f} @3px {m('bf3'):.3f} | Boundary-IoU {m('biou'):.3f}")


if __name__ == "__main__":
    main()
