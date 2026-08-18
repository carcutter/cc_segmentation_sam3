#!/usr/bin/env python3
"""
Find the best 'plausible fill' to make predicted antenna masks continuous.

Runs the antenna UNet once on test antenna imgs, then compares fill strategies:
  raw            : no fill
  vclose121      : vertical morphological close, tall kernel (bridge <=121px gaps)
  colfill        : per-column vertical-span fill — for each column with antenna
                   pixels, fill from its top to bottom antenna pixel. Connects
                   vertically-stacked fragments (whip base->tip) without merging
                   separate antennas at different x. Capped per-column gap to avoid
                   absurd fills.
Reports %fragmented, mean components, white IoU, white recall for each.

Run from repo root:
  PYTHONPATH=. /home/rutger/miniconda3/envs/cc_sam3/bin/python carcutter/car_bbox_detector/antenna_continuity.py --n 290
"""
import argparse, csv
import numpy as np, cv2
from PIL import Image
import torch, segmentation_models_pytorch as smp
from carcutter.car_bbox_detector.seg_eval import unet_full_mask, mask_iou
WHITE = (255, 255, 255)


def ncomp(m): n, _ = cv2.connectedComponents(m.astype(np.uint8)); return n-1
def recall(g, p): d = g.sum(); return (g & p).sum()/d if d else 1.0


def vclose(m, gap):
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, gap))
    return cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, k) > 0


def colfill(m, max_gap=400):
    """Per-column: fill from top to bottom antenna pixel (bridges vertical gaps)."""
    out = m.copy()
    ys, xs = np.where(m)
    if len(xs) == 0: return out
    for x in np.unique(xs):
        col = np.where(m[:, x])[0]
        if col[-1]-col[0] <= max_gap:
            out[col[0]:col[-1]+1, x] = True
    return out


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
    strats = {"raw": lambda m: m, "vclose121": lambda m: vclose(m, 121), "colfill": colfill}
    agg = {k: {"nc": [], "frag": [], "iou": [], "rec": []} for k in strats}
    n = 0
    for r in rows:
        w = (np.array(Image.open(r["mask"]).convert("RGB")) == WHITE).all(2)
        if w.sum() < 20: continue
        n += 1
        pred = unet_full_mask(model, Image.open(r["image"]).convert("RGB"), w.astype(np.uint8)*255, args.crop_size, device)
        for k, fn in strats.items():
            p = fn(pred); a = agg[k]
            a["nc"].append(ncomp(p)); a["frag"].append(ncomp(p) > 1)
            a["iou"].append(mask_iou(w, p)); a["rec"].append(recall(w, p))
    m = lambda x: float(np.mean(x))
    print(f"Antenna continuity fills on {n} test imgs\n")
    print(f"  {'strategy':<12}{'%frag':>8}{'mean_cc':>9}{'IoU':>8}{'recall':>8}")
    for k in strats:
        a = agg[k]
        print(f"  {k:<12}{100*m(a['frag']):>7.1f}%{m(a['nc']):>9.2f}{m(a['iou']):>8.3f}{m(a['rec']):>8.3f}")


if __name__ == "__main__":
    main()
