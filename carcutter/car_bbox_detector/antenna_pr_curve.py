#!/usr/bin/env python3
"""
Antenna precision/recall curve vs the DETECTION threshold (RF-DETR boxes -> antenna UNet).
Pixel-level, micro-averaged over ALL test images (so FP firing on no-antenna cars counts
against precision), with a 3px tolerance band (antennas are thin).

Sweeps the detector antenna-confidence threshold; at each t the antenna mask = union of
colfill(UNet) over boxes with conf>=t. Outputs PR points (csv) + PR plot + AP.

Run from repo root:
  PYTHONPATH=. python carcutter/car_bbox_detector/antenna_pr_curve.py --n 300
"""
import argparse, csv
from pathlib import Path
import numpy as np, cv2, torch
from PIL import Image
import segmentation_models_pytorch as smp
import rfdetr
from carcutter.car_bbox_detector.gen_coarse_masks import coarse_prob
from carcutter.car_bbox_detector.antenna_continuity import colfill
WHITE = (255, 255, 255); ANT = 1
EXP = "carcutter/car_bbox_detector/experiments"
THRS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80]


def load(ck, dev):
    m = smp.Unet("efficientnet-b4", encoder_weights=None, in_channels=3, classes=1).to(dev).eval()
    st = torch.load(ck, map_location=dev); m.load_state_dict(st.get("model", st)); return m


def boxmask(b, H, W):
    m = np.zeros((H, W), np.uint8); x0, y0, x1, y1 = [int(v) for v in b]
    m[max(0, y0):min(H, y1 + 1), max(0, x0):min(W, x1 + 1)] = 255; return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="carcutter/car_bbox_detector/data/index.csv")
    ap.add_argument("--out", default="carcutter/car_bbox_detector/data/antenna_pr.csv")
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()
    dev = "cuda"
    det = rfdetr.RFDETRMedium.from_checkpoint(f"{EXP}/rfdetr_car_antenna_v1/checkpoint_best_total.pth")
    model = load(f"{EXP}/unet_antenna_v1/checkpoints/best.pt", dev)
    rows = [r for r in csv.DictReader(open(args.index)) if r["split"] == "test"][:args.n]
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    acc = {t: {"corr": 0, "pred": 0, "cov": 0} for t in THRS}
    white_total = 0
    for i, r in enumerate(rows):
        pil = Image.open(r["image"]).convert("RGB"); img = np.array(pil); H, W = img.shape[:2]
        white = (np.array(Image.open(r["mask"]).convert("RGB")) == WHITE).all(2)
        white_total += int(white.sum())
        wdil = cv2.dilate(white.astype(np.uint8), k3) > 0
        d = det.predict(pil, threshold=0.05)
        cls, conf, xyxy = np.asarray(d.class_id), np.asarray(d.confidence), np.asarray(d.xyxy)
        boxes = [(xyxy[j], float(conf[j])) for j in range(len(xyxy)) if cls[j] == ANT and conf[j] >= 0.05]
        bmask = {}                                   # per-box mask, computed once
        for bi, (b, c) in enumerate(boxes):
            bmask[bi] = colfill(coarse_prob(model, img, boxmask(b, H, W), 256, dev) > 0.5)
        for t in THRS:
            pred = np.zeros((H, W), bool)
            for bi, (b, c) in enumerate(boxes):
                if c >= t:
                    pred |= bmask[bi]
            ps = int(pred.sum())
            acc[t]["pred"] += ps
            if ps:
                acc[t]["corr"] += int((pred & wdil).sum())
                pdil = cv2.dilate(pred.astype(np.uint8), k3) > 0
                acc[t]["cov"] += int((white & pdil).sum())
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)

    pts = []
    for t in THRS:
        a = acc[t]
        P = a["corr"] / a["pred"] if a["pred"] else 1.0
        R = a["cov"] / white_total if white_total else 0.0
        pts.append((t, P, R))
    # AP = area under P-R (sort by recall)
    srt = sorted([(R, P) for _, P, R in pts])
    ap_area = 0.0
    for j in range(1, len(srt)):
        ap_area += (srt[j][0] - srt[j - 1][0]) * (srt[j][1] + srt[j - 1][1]) / 2
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["method", "thr", "precision", "recall"])
        for t, P, R in pts:
            w.writerow(["ours", f"{t:.2f}", f"{P:.4f}", f"{R:.4f}"])
    print(f"\nOURS antenna PR (pixel, 3px tol, micro over {len(rows)} imgs):")
    print(f"  {'thr':>5}{'precision':>11}{'recall':>9}")
    for t, P, R in pts:
        print(f"  {t:>5.2f}{P:>11.3f}{R:>9.3f}")
    print(f"  PR-AUC(area) ~ {ap_area:.3f}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
