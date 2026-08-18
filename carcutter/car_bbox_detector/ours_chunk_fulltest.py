#!/usr/bin/env python3
"""
ours-v2 outline 'cut-off chunk' over the FULL test set (symmetric to the prod full-test
estimate). Lean: BiRefNet body ∪ antenna(det thr 0.5)+colfill, keep_main FP-clean. No
holes/refiners (outline only). Reports the segment-drop distribution vs GT union.

Run from repo root:
  PYTHONPATH=.:carcutter/car_bbox_detector/birefnet/BiRefNet python carcutter/car_bbox_detector/ours_chunk_fulltest.py
"""
import argparse, csv
from pathlib import Path
import numpy as np, cv2, torch
from PIL import Image
import segmentation_models_pytorch as smp
import rfdetr
from carcutter.car_bbox_detector.gen_coarse_masks import coarse_prob
from carcutter.car_bbox_detector.antenna_continuity import colfill
from carcutter.car_bbox_detector.antenna_fp_clean import keep_main
from carcutter.car_bbox_detector.birefnet_eval import load_birefnet, birefnet_outline
from carcutter.car_bbox_detector.cascade_eval import predict_union
WHITE = (255, 255, 255); ANT = 1
EXP = "carcutter/car_bbox_detector/experiments"
BF = "carcutter/car_bbox_detector/birefnet/BiRefNet/ckpts/car_outline_v1/epoch_10.pth"
ANT_THR = 0.5


def load_unet(ck, dev):
    m = smp.Unet("efficientnet-b4", encoder_weights=None, in_channels=3, classes=1).to(dev).eval()
    st = torch.load(ck, map_location=dev); m.load_state_dict(st.get("model", st)); return m


def boxmask(b, H, W):
    m = np.zeros((H, W), np.uint8); x0, y0, x1, y1 = [int(v) for v in b]
    m[max(0, y0):min(H, y1 + 1), max(0, x0):min(W, x1 + 1)] = 255; return m


def maxchunk(gt, pred):
    miss = (gt & ~pred).astype(np.uint8)
    if miss.sum() == 0: return 0.0
    n, _, st, _ = cv2.connectedComponentsWithStats(miss, 8)
    return float(st[1:, cv2.CC_STAT_AREA].max()) / float(gt.sum()) if n > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="carcutter/car_bbox_detector/data/index.csv")
    ap.add_argument("--n", type=int, default=10000)
    args = ap.parse_args()
    dev = "cuda"
    det = rfdetr.RFDETRMedium.from_checkpoint(f"{EXP}/rfdetr_car_antenna_v1/checkpoint_best_total.pth")
    bf = load_birefnet(BF, dev)
    ant = load_unet(f"{EXP}/unet_antenna_v1/checkpoints/best.pt", dev)
    rows = [r for r in csv.DictReader(open(args.index)) if r["split"] == "test"][:args.n]
    chunks = []
    for i, r in enumerate(rows):
        pil = Image.open(r["image"]).convert("RGB"); img = np.array(pil); H, W = img.shape[:2]
        gt = np.array(Image.open(r["mask"]).convert("RGB")).sum(2) > 0
        if gt.sum() < 20: continue
        carbox = predict_union(det, pil, 0.3, 0.15)
        if carbox is None: continue
        body = birefnet_outline(bf, img, carbox, dev)
        am = np.zeros((H, W), bool)
        d = det.predict(pil, threshold=ANT_THR)
        cls, conf, xyxy = np.asarray(d.class_id), np.asarray(d.confidence), np.asarray(d.xyxy)
        for j in range(len(xyxy)):
            if cls[j] == ANT and conf[j] >= ANT_THR:
                am |= colfill(coarse_prob(ant, img, boxmask(xyxy[j], H, W), 256, dev) > 0.5)
        outline = keep_main(body | am, 9)
        chunks.append(maxchunk(gt, outline))
        if (i + 1) % 100 == 0: print(f"  {i+1}/{len(rows)}", flush=True)
    c = np.array(chunks); N = len(c)
    print(f"\nOURS-v2 outline 'cut-off chunk' over FULL test set, N={N}:")
    print(f"  mean {c.mean():.4f}  p90 {np.quantile(c,.9):.4f}  p99 {np.quantile(c,.99):.4f}  max {c.max():.4f}")
    for t in (0.02, 0.05, 0.10, 0.20):
        k = int((c > t).sum()); print(f"  drops > {int(t*100):>2}% of car: {k:>3}/{N} = {100*k/N:.1f}%")


if __name__ == "__main__":
    main()
