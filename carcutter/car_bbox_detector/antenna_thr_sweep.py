#!/usr/bin/env python3
"""
Antenna DETECTION-threshold sweep (the FP lever). FPs come from the recall-first
detector (thr 0.15) proposing antenna boxes on stick-like structures; the UNet then
fires inside them. Raising the threshold drops low-confidence FP boxes. Measures the
recall vs FP trade-off so we can pick a deploy threshold.

Per image: one detector pass (thr 0.15), then assemble antenna mask using boxes with
conf>=thr for each candidate thr. v1 antenna UNet.

Run from repo root:
  PYTHONPATH=. python carcutter/car_bbox_detector/antenna_thr_sweep.py --n 300
"""
import argparse, csv
import numpy as np, cv2, torch
from PIL import Image
import segmentation_models_pytorch as smp
import rfdetr
from carcutter.car_bbox_detector.gen_coarse_masks import coarse_prob
from carcutter.car_bbox_detector.antenna_continuity import colfill
from carcutter.car_bbox_detector.seg_eval import mask_iou
WHITE = (255, 255, 255); ANT = 1
EXP = "carcutter/car_bbox_detector/experiments"
THRS = [0.15, 0.25, 0.35, 0.50]


def load(ck, dev):
    m = smp.Unet("efficientnet-b4", encoder_weights=None, in_channels=3, classes=1).to(dev).eval()
    st = torch.load(ck, map_location=dev); m.load_state_dict(st.get("model", st)); return m


def boxmask(b, H, W):
    m = np.zeros((H, W), np.uint8); x0, y0, x1, y1 = [int(v) for v in b]
    m[max(0, y0):min(H, y1 + 1), max(0, x0):min(W, x1 + 1)] = 255; return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="carcutter/car_bbox_detector/data/index.csv")
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()
    dev = "cuda"
    det = rfdetr.RFDETRMedium.from_checkpoint(f"{EXP}/rfdetr_car_antenna_v1/checkpoint_best_total.pth")
    model = load(f"{EXP}/unet_antenna_v1/checkpoints/best.pt", dev)
    rows = [r for r in csv.DictReader(open(args.index)) if r["split"] == "test"][:args.n]

    agg = {t: {"rec": [], "fire_noant": []} for t in THRS}
    for i, r in enumerate(rows):
        pil = Image.open(r["image"]).convert("RGB"); img = np.array(pil); H, W = img.shape[:2]
        white = (np.array(Image.open(r["mask"]).convert("RGB")) == WHITE).all(2)
        has_ant = white.sum() >= 20
        d = det.predict(pil, threshold=0.15)
        cls, conf, xyxy = np.asarray(d.class_id), np.asarray(d.confidence), np.asarray(d.xyxy)
        ant = [(xyxy[j], conf[j]) for j in range(len(xyxy)) if cls[j] == ANT and conf[j] >= 0.15]
        # cache per-box masks once
        boxpred = {}
        for t in THRS:
            pred = np.zeros((H, W), bool)
            for bi, (b, c) in enumerate(ant):
                if c < t:
                    continue
                if bi not in boxpred:
                    boxpred[bi] = colfill(coarse_prob(model, img, boxmask(b, H, W), 256, dev) > 0.5)
                pred |= boxpred[bi]
            if has_ant:
                agg[t]["rec"].append((white & pred).sum() / white.sum())
            else:
                agg[t]["fire_noant"].append(1.0 if pred.sum() >= 20 else 0.0)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)

    m = lambda a: float(np.mean(a)) if a else float("nan")
    print(f"\nAntenna detection-threshold sweep (v1), n={len(rows)}:\n")
    print(f"  {'thr':>5}{'recall':>9}{'fire_noAnt%':>13}")
    for t in THRS:
        a = agg[t]
        print(f"  {t:>5.2f}{m(a['rec']):>9.3f}{100*m(a['fire_noant']):>12.1f}%")
    print(f"\n(recall on {len(agg[0.15]['rec'])} antenna imgs; fire_noAnt on "
          f"{len(agg[0.15]['fire_noant'])} no-antenna imgs)")


if __name__ == "__main__":
    main()
