#!/usr/bin/env python3
"""
Classifier-gated antenna PR: detector boxes at a FIXED low threshold (high recall),
then keep a box's UNet mask only if the discriminator score >= c. Sweep c -> PR curve.
If this curve sits ABOVE the ours-threshold curve, the discriminator adds real value.

Run from repo root:
  PYTHONPATH=. python carcutter/car_bbox_detector/antenna_clf_pr.py --n 300
"""
import argparse, csv
from pathlib import Path
import numpy as np, cv2, torch, timm
import segmentation_models_pytorch as smp
import rfdetr
from PIL import Image
from carcutter.car_bbox_detector.gen_coarse_masks import coarse_prob
from carcutter.car_bbox_detector.antenna_continuity import colfill
WHITE = (255, 255, 255); ANT = 1
EXP = "carcutter/car_bbox_detector/experiments"
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
DET_THR = 0.10                       # fixed low detection threshold (recall); classifier does precision
CLF_THRS = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def load_unet(ck, dev):
    m = smp.Unet("efficientnet-b4", encoder_weights=None, in_channels=3, classes=1).to(dev).eval()
    st = torch.load(ck, map_location=dev); m.load_state_dict(st.get("model", st)); return m


def boxmask(b, H, W):
    m = np.zeros((H, W), np.uint8); x0, y0, x1, y1 = [int(v) for v in b]
    m[max(0, y0):min(H, y1 + 1), max(0, x0):min(W, x1 + 1)] = 255; return m


def square_crop(img, box, pad=0.20):
    H, W = img.shape[:2]; x0, y0, x1, y1 = [float(v) for v in box]
    bw, bh = x1 - x0, y1 - y0; px, py = bw * pad, bh * pad
    x0, y0 = max(0, int(x0 - px)), max(0, int(y0 - py)); x1, y1 = min(W, int(x1 + px)), min(H, int(y1 + py))
    c = img[y0:y1, x0:x1]
    if c.shape[0] < 6 or c.shape[1] < 6: return None
    side = max(c.shape[:2]); xo, yo = (side - c.shape[1]) // 2, (side - c.shape[0]) // 2
    sq = np.zeros((side, side, 3), np.uint8); sq[yo:yo + c.shape[0], xo:xo + c.shape[1]] = c
    return sq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="carcutter/car_bbox_detector/data/index.csv")
    ap.add_argument("--out", default="carcutter/car_bbox_detector/data/antenna_pr_clf.csv")
    ap.add_argument("--n", type=int, default=300); args = ap.parse_args()
    dev = "cuda"
    det = rfdetr.RFDETRMedium.from_checkpoint(f"{EXP}/rfdetr_car_antenna_v1/checkpoint_best_total.pth")
    unet = load_unet(f"{EXP}/unet_antenna_v1/checkpoints/best.pt", dev)
    clf = timm.create_model("efficientnet_b0", pretrained=False, num_classes=1).to(dev).eval()
    clf.load_state_dict(torch.load(f"{EXP}/antenna_clf_v1/best.pt", map_location=dev)["model"])
    rows = [r for r in csv.DictReader(open(args.index)) if r["split"] == "test"][:args.n]
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    acc = {t: {"corr": 0, "pred": 0, "cov": 0} for t in CLF_THRS}; white_total = 0
    for i, r in enumerate(rows):
        pil = Image.open(r["image"]).convert("RGB"); img = np.array(pil); H, W = img.shape[:2]
        white = (np.array(Image.open(r["mask"]).convert("RGB")) == WHITE).all(2)
        white_total += int(white.sum()); wdil = cv2.dilate(white.astype(np.uint8), k3) > 0
        d = det.predict(pil, threshold=DET_THR)
        cls, conf, xyxy = np.asarray(d.class_id), np.asarray(d.confidence), np.asarray(d.xyxy)
        boxes = [xyxy[j] for j in range(len(xyxy)) if cls[j] == ANT and conf[j] >= DET_THR]
        masks, scores = [], []
        for b in boxes:
            sq = square_crop(img, b)
            if sq is None: continue
            inp = cv2.resize(sq, (224, 224))
            t = torch.from_numpy(((inp / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]).float().to(dev)
            with torch.no_grad():
                cs = float(torch.sigmoid(clf(t)).item())
            masks.append(colfill(coarse_prob(unet, img, boxmask(b, H, W), 256, dev) > 0.5)); scores.append(cs)
        for t in CLF_THRS:
            pred = np.zeros((H, W), bool)
            for m, s in zip(masks, scores):
                if s >= t: pred |= m
            ps = int(pred.sum()); acc[t]["pred"] += ps
            if ps:
                acc[t]["corr"] += int((pred & wdil).sum())
                acc[t]["cov"] += int((white & (cv2.dilate(pred.astype(np.uint8), k3) > 0)).sum())
        if (i + 1) % 50 == 0: print(f"  {i+1}/{len(rows)}", flush=True)

    pts = [(t, acc[t]["corr"] / acc[t]["pred"] if acc[t]["pred"] else 1.0,
            acc[t]["cov"] / white_total if white_total else 0.0) for t in CLF_THRS]
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["method", "thr", "precision", "recall"])
        for t, P, R in pts: w.writerow(["clf-gate", f"{t:.2f}", f"{P:.4f}", f"{R:.4f}"])
    print(f"\nClassifier-gated antenna PR (det_thr={DET_THR} fixed, sweep clf score):")
    print(f"  {'clf':>5}{'precision':>11}{'recall':>9}")
    for t, P, R in pts: print(f"  {t:>5.2f}{P:>11.3f}{R:>9.3f}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
