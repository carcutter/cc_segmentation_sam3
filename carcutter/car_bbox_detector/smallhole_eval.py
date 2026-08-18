#!/usr/bin/env python3
"""
Eval the dedicated small-hole pipeline: run the small-hole UNet on the hole-prone region
tiles (roof/lowL/lowR of the car bbox), upsampled to 512, stitch its holes, UNION with the
existing punchout (from the 1536 body matte, cached in e2e_v3 d['punchout']), and re-count
small structural-hole instance recall vs GT. Compares: punchout-only vs +dedicated vs prod.

Run from repo root:
  PYTHONPATH=. python carcutter/car_bbox_detector/smallhole_eval.py
"""
import argparse, csv, os
from pathlib import Path
import numpy as np, cv2, torch
import segmentation_models_pytorch as smp
from PIL import Image
from scipy.ndimage import binary_fill_holes
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
EXP = "carcutter/car_bbox_detector/experiments/unet_smallhole_v1"
SMALL_FRAC = 0.006; MINPX = 25; TILE = 512
REGIONS = {"roof": (0.0, 0.0, 1.0, 0.30), "lowL": (0.0, 0.48, 0.62, 1.0), "lowR": (0.38, 0.48, 1.0, 1.0)}


def binar(p):
    if not p or not os.path.exists(p): return None
    a = np.array(Image.open(p)); return (a.sum(2) > 0) if a.ndim == 3 else (a > 0)


def small_comps(holemask, car_area):
    n, lbl, st, _ = cv2.connectedComponentsWithStats(holemask.astype(np.uint8), 8)
    return [(lbl == i, st[i, cv2.CC_STAT_AREA]) for i in range(1, n)
            if MINPX <= st[i, cv2.CC_STAT_AREA] < SMALL_FRAC * car_area]


@torch.no_grad()
def run_regions(model, img, carbox, dev, thr=0.5):
    H, W = img.shape[:2]; x0, y0, x1, y1 = [int(v) for v in carbox]
    x0, y0 = max(0, x0), max(0, y0); x1, y1 = min(W, x1), min(H, y1)
    bw, bh = x1 - x0, y1 - y0
    out = np.zeros((H, W), np.float32)
    for (fx0, fy0, fx1, fy1) in REGIONS.values():
        rx0, ry0 = int(x0 + fx0 * bw), int(y0 + fy0 * bh); rx1, ry1 = int(x0 + fx1 * bw), int(y0 + fy1 * bh)
        if rx1 - rx0 < 16 or ry1 - ry0 < 16: continue
        tile = cv2.resize(img[ry0:ry1, rx0:rx1], (TILE, TILE))
        t = torch.from_numpy(((tile / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]).float().to(dev)
        pr = torch.sigmoid(model(t))[0, 0].cpu().numpy()
        pr = cv2.resize(pr, (rx1 - rx0, ry1 - ry0))
        out[ry0:ry1, rx0:rx1] = np.maximum(out[ry0:ry1, rx0:rx1], pr)
    return out   # soft prob map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=f"{EXP}/best.pt")
    ap.add_argument("--e2e", default="carcutter/car_bbox_detector/data/e2e_v3")
    ap.add_argument("--index", default="carcutter/car_bbox_detector/data/index.csv")
    args = ap.parse_args()
    dev = "cuda"
    idx = {Path(r["image"]).stem: r for r in csv.DictReader(open(args.index))}
    model = smp.Unet("efficientnet-b2", encoder_weights=None, in_channels=3, classes=1).to(dev).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location=dev)["model"])
    THRS = [0.5, 0.7, 0.85, 0.92, 0.97]
    gt_n = 0; po_hit = 0; prod_hit = 0
    aug_hit = {t: 0 for t in THRS}
    prec_po = [0, 0]; prec_aug = {t: [0, 0] for t in THRS}
    for npz in sorted((Path(args.e2e) / "masks").glob("*.npz")):
        r = idx.get(npz.stem)
        if r is None: continue
        d = np.load(npz)
        if "carbox" not in d or "punchout" not in d: continue
        rgb = np.array(Image.open(r["mask"]).convert("RGB")); union = rgb.sum(2) > 0
        ca = float(union.sum())
        if ca < 1500: continue
        img = np.array(Image.open(r["image"]).convert("RGB"))
        allholes = binary_fill_holes(union) & ~union
        gt_small = small_comps(allholes, ca)
        po = d["punchout"].astype(bool)
        soft = run_regions(model, img, d["carbox"], dev)
        prod = binar(r["prod_outline"].replace("/outline/", "/holes_punchout/")) if (r["prod_outline"] and os.path.exists(r["prod_outline"])) else None
        for comp, area in gt_small:
            gt_n += 1
            if (comp & po).sum() >= 0.3 * area: po_hit += 1
            if prod is not None and (comp & prod).sum() >= 0.3 * area: prod_hit += 1
            for t in THRS:
                if (comp & (po | (soft > t))).sum() >= 0.3 * area: aug_hit[t] += 1
        for comp, area in small_comps(po, ca):
            prec_po[1] += 1; prec_po[0] += int((comp & allholes).sum() >= 0.3 * area)
        for t in THRS:
            for comp, area in small_comps(po | (soft > t), ca):
                prec_aug[t][1] += 1; prec_aug[t][0] += int((comp & allholes).sum() >= 0.3 * area)
    g = gt_n
    print(f"\nSMALL hole recall/precision (GT small holes={g}):")
    print(f"  punchout only : recall {100*po_hit/g:.0f}%  precision {100*prec_po[0]/max(1,prec_po[1]):.0f}%")
    print(f"  prod          : recall {100*prod_hit/g:.0f}%")
    print(f"  + dedicated UNet (punchout ∪ dedicated>thr):")
    print(f"    {'thr':>5}{'recall':>9}{'precision':>11}")
    for t in THRS:
        print(f"    {t:>5.2f}{100*aug_hit[t]/g:>8.0f}%{100*prec_aug[t][0]/max(1,prec_aug[t][1]):>10.0f}%")


if __name__ == "__main__":
    main()
