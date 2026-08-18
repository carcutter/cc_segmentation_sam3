#!/usr/bin/env python3
"""
RE-BASELINE vs the ACTUAL shipped production masks (masks_prod) — nothing of ours has shipped, so
masks_prod is the true starting point. Outline BF@1/IoU by view + ANTENNA COVERAGE by view for:
  realprod (masks_prod) | v2sq (car_outline_v2, our square, unshipped) | bucket5 (our 5-bucket).
Antenna coverage = recall of GT antenna (white) pixels falling inside the predicted outline silhouette
(prod has no antenna class -> only what its outline happens to include). Tests whether bucketing
improved antenna inclusion -> informs a 3rd antenna head.

Run: PYTHONPATH=.:carcutter/car_bbox_detector/birefnet/BiRefNet HF_HOME=... \
  /home/rutger/miniconda3/envs/cc_sam3/bin/python carcutter/car_bbox_detector/measure_vs_realprod.py --n 930
"""
import argparse, csv, os, sys
from collections import defaultdict
import numpy as np, torch
from PIL import Image
sys.path.insert(0, "carcutter/car_bbox_detector/birefnet/BiRefNet")
from carcutter.car_bbox_detector.seg_eval import boundary_f, mask_iou
from carcutter.car_bbox_detector.birefnet_eval import load_birefnet, birefnet_outline
from carcutter.car_bbox_detector.probe_birefnet_aspect import route, outline_bucketed

ROOT = "carcutter/car_bbox_detector"; EXP = f"{ROOT}/experiments"
V2 = f"{ROOT}/birefnet/BiRefNet/ckpts/car_outline_v2/epoch_16.pth"
B5 = f"{EXP}/birefnet_aspect_prod_bucket5.pt"
WHITE = (255, 255, 255)
VIEWS = ["straight", "corner", "corner34", "side", "other", "ALL"]


def vg(b):
    import re; b = b.lower()
    if "side" in b: return "side"
    if "trunk" in b or "interior" in b or "engine" in b: return "other"
    if re.search(r"(34|3-4|quarter)", b): return "corner34"
    fr = ("front" in b) or ("rear" in b) or ("back" in b); lr = ("left" in b) or ("right" in b)
    if fr and lr: return "corner"
    if fr: return "straight"
    return "other"


def load_b5(dev):
    from models.birefnet import BiRefNet
    from utils import check_state_dict
    raw = torch.load(B5, map_location="cpu", weights_only=False)
    m = BiRefNet(bb_pretrained=False).to(dev).eval()
    m.load_state_dict(check_state_dict(raw["model"])); return m


def binar(p):
    a = np.array(Image.open(p)); return (a.sum(2) > 0) if a.ndim == 3 else (a > 0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=930)
    ap.add_argument("--index", default=f"{ROOT}/data/index.csv"); args = ap.parse_args(); dev = "cuda"
    v2 = load_birefnet(V2, dev); b5 = load_b5(dev)
    rows = [r for r in csv.DictReader(open(args.index)) if r["split"] == "test"][:args.n]

    bf = {m: defaultdict(list) for m in ("realprod", "v2sq", "bucket5")}
    iou = {m: defaultdict(list) for m in ("realprod", "v2sq", "bucket5")}
    antcov = {m: defaultdict(list) for m in ("realprod", "v2sq", "bucket5")}
    for i, r in enumerate(rows):
        po = r["prod_outline"]
        if not (po and os.path.exists(po)): continue
        img = np.array(Image.open(r["image"]).convert("RGB"))
        rgb = np.array(Image.open(r["mask"]).convert("RGB")); gt = rgb.sum(2) > 0; white = (rgb == WHITE).all(2)
        if gt.sum() < 20: continue
        v = vg(os.path.basename(r["image"]))
        bw, bh = int(r["bw"]), int(r["bh"]); box = (int(r["x"]), int(r["y"]), int(r["x"]) + bw, int(r["y"]) + bh)
        pred = {"realprod": binar(po),
                "v2sq": birefnet_outline(v2, img, box, dev, size=1024),
                "bucket5": outline_bucketed(b5, img, box, route(bw / bh), dev)}
        for m, p in pred.items():
            for vv in (v, "ALL"):
                bf[m][vv].append(boundary_f(gt, p, 1)); iou[m][vv].append(mask_iou(gt, p))
            if white.sum() >= 20:
                cov = (white & p).sum() / white.sum()
                antcov[m][v].append(cov); antcov[m]["ALL"].append(cov)
        if (i + 1) % 100 == 0: print(f"  {i+1}/{len(rows)}", flush=True)

    def table(title, agg, fmt):
        print(f"\n=== {title} ===")
        print(f"  {'view':<9}{'n':>5}{'realprod':>10}{'v2sq':>9}{'bucket5':>9}")
        for vv in VIEWS:
            if agg['realprod'][vv]:
                n = len(agg['realprod'][vv])
                print(f"  {vv:<9}{n:>5}" + "".join(fmt(np.mean(agg[m][vv])) for m in ("realprod","v2sq","bucket5")))
    f3 = lambda x: f"{x:>9.3f}"
    table("OUTLINE BF@1 by view (vs GT)", bf, f3)
    table("OUTLINE IoU by view", iou, f3)
    table("ANTENNA coverage by view (recall of GT antenna inside outline)", antcov, f3)


if __name__ == "__main__":
    main()
