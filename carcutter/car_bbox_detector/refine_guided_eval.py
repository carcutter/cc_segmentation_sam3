#!/usr/bin/env python3
"""
Prototype #1 (training-free): guided-filter edge-snap of the coarse UNet mask.

The coarse 768 UNet probability map (back-projected to native res) is blurry at
the boundary. A guided filter (He et al.) with the native grayscale image as
guide pulls the mask probability toward strong image edges (the car silhouette)
before thresholding — a cheap native-res sharpening with no training.

Reports COARSE (UNet 768) vs REFINED (guided) vs PROD on boundary metrics.
If REFINED clears prod's Boundary-F@1px=0.708, we got a free win; else we train
the boundary-refinement UNet (prototype #2).

Run from repo root:
  PYTHONPATH=. /home/rutger/miniconda3/envs/cc_sam3/bin/python carcutter/car_bbox_detector/refine_guided_eval.py \
      --weights carcutter/car_bbox_detector/experiments/unet_car_seg_v1/checkpoints/best.pt --n 200
"""
import argparse, csv
import numpy as np, cv2, torch
from PIL import Image
import segmentation_models_pytorch as smp
from carcutter.unet.dataset import _crop_params_from_gt
from carcutter.car_bbox_detector.seg_eval import boundary_f, boundary_iou, mask_iou, MEAN, STD


def box(x, r): return cv2.boxFilter(x, -1, (r, r), borderType=cv2.BORDER_REFLECT)

def guided_filter(I, p, r=8, eps=1e-4):
    """I: guide gray float[0,1] HxW, p: input prob HxW -> edge-aware smoothed p."""
    mI, mp = box(I, r), box(p, r)
    cII, cIp = box(I*I, r), box(I*p, r)
    vI = cII - mI*mI; cov = cIp - mI*mp
    a = cov/(vI+eps); b = mp - a*mI
    return box(a, r)*I + box(b, r)


@torch.no_grad()
def coarse_prob(model, pil, gt_mask, crop_size, device):
    p = _crop_params_from_gt(gt_mask, 1.5); H, W = gt_mask.shape
    top, left, ch, cw = p if p else (0, 0, H, W)
    crop = np.array(pil)[top:top+ch, left:left+cw]
    side = max(ch, cw); xo, yo = (side-cw)//2, (side-ch)//2
    sq = np.zeros((side, side, 3), np.uint8); sq[yo:yo+ch, xo:xo+cw] = crop
    inp = cv2.resize(sq, (crop_size, crop_size))
    t = torch.from_numpy(((inp/255.0-MEAN)/STD).transpose(2,0,1)[None]).float().to(device)
    pr = torch.sigmoid(model(t)[0,0]).cpu().numpy()
    pr = cv2.resize(pr, (side, side))[yo:yo+ch, xo:xo+cw]
    full = np.zeros((H, W), np.float32); full[top:top+ch, left:left+cw] = pr
    return full


def prod_bbox_mask(path):
    a = np.array(Image.open(path)); return a > 0 if a.ndim == 2 else a.sum(2) > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="carcutter/car_bbox_detector/experiments/unet_car_seg_v1/checkpoints/best.pt")
    ap.add_argument("--index", default="carcutter/car_bbox_detector/data/index.csv")
    ap.add_argument("--crop-size", type=int, default=768)
    ap.add_argument("--radius", type=int, default=8)
    ap.add_argument("--eps", type=float, default=1e-4)
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = smp.Unet(encoder_name="efficientnet-b4", encoder_weights=None, in_channels=3, classes=1).to(device).eval()
    st = torch.load(args.weights, map_location=device); model.load_state_dict(st.get("model", st))

    rows = [r for r in csv.DictReader(open(args.index)) if r["split"] == "test"][:args.n]
    agg = {"coarse": {}, "refined": {}, "prod": {}}
    def add(d,k,v): d[k]=d.get(k,[])+[v]
    for r in rows:
        gt = np.array(Image.open(r["mask"]).convert("RGB")).sum(2) > 0
        img = np.array(Image.open(r["image"]).convert("RGB"))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)/255.0
        prob = coarse_prob(model, Image.fromarray(img), gt.astype(np.uint8), args.crop_size, device)
        refined = guided_filter(gray, prob, args.radius, args.eps) > 0.5
        coarse = prob > 0.5
        masks = {"coarse": coarse, "refined": refined}
        if r["prod_outline"]: masks["prod"] = prod_bbox_mask(r["prod_outline"])
        for nm, pr in masks.items():
            add(agg[nm], "iou", mask_iou(gt, pr))
            for t in (1,2,3): add(agg[nm], f"bf{t}", boundary_f(gt, pr, t))
            add(agg[nm], "biou", boundary_iou(gt, pr, 2))
    print(f"Guided-filter refinement on {len(rows)} imgs (r={args.radius}, eps={args.eps})\n")
    for nm in ("coarse","refined","prod"):
        a=agg[nm];
        if not a: continue
        m=lambda k: np.mean(a[k])
        print(f"{nm:8}: IoU {m('iou'):.4f} | Boundary-F @1px {m('bf1'):.3f} @2px {m('bf2'):.3f} @3px {m('bf3'):.3f} | Boundary-IoU {m('biou'):.3f}")


if __name__ == "__main__":
    main()
