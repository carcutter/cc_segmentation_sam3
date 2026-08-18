#!/usr/bin/env python3
"""
MEASURE (no retrain): does the bucketed body silhouette (prod_bucket5) capture small punchout holes
at its native res as well as the dedicated 1536-square pass? If yes -> we can DROP the 1536 pass
(37% of pipeline compute) in the bucketed world. Structural-hole recall by size, vs GT topological
holes, for: car_outline_v2@1024(sq) | car_outline_v2@1536(sq) | prod_bucket5(bucketed).

Read-only inference on the test split. Run from repo root:
  PYTHONPATH=.:carcutter/car_bbox_detector/birefnet/BiRefNet HF_HOME=... \
  /home/rutger/miniconda3/envs/cc_sam3/bin/python carcutter/car_bbox_detector/measure_bucket_holes.py --n 300
"""
import argparse, csv, os, math
from pathlib import Path
import numpy as np, cv2, torch
from PIL import Image
from scipy.ndimage import binary_fill_holes
from carcutter.car_bbox_detector.birefnet_eval import load_birefnet, birefnet_outline
from carcutter.car_bbox_detector.probe_birefnet_aspect import BUCKETS, CENTERS, route, outline_bucketed
from carcutter.car_bbox_detector.antenna_fp_clean import keep_main

ROOT = "carcutter/car_bbox_detector"; EXP = f"{ROOT}/experiments"
V2 = f"{ROOT}/birefnet/BiRefNet/ckpts/car_outline_v2/epoch_16.pth"
B5 = f"{EXP}/birefnet_aspect_prod_bucket5.pt"
MINPX = 25
SIZES = [("small", 0, 0.005), ("med", 0.005, 0.02), ("large", 0.02, 1.0)]


def load_b5(dev):  # prod_bucket5 is wrapped {"model","epoch"}
    import sys; sys.path.insert(0, f"{ROOT}/birefnet/BiRefNet")
    from models.birefnet import BiRefNet
    from utils import check_state_dict
    raw = torch.load(B5, map_location="cpu", weights_only=False)
    m = BiRefNet(bb_pretrained=False).to(dev).eval()
    m.load_state_dict(check_state_dict(raw["model"])); return m


def comps(mask, ca, lo, hi):
    n, lbl, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return [(lbl == i, st[i, cv2.CC_STAT_AREA]) for i in range(1, n)
            if MINPX <= st[i, cv2.CC_STAT_AREA] and lo <= st[i, cv2.CC_STAT_AREA] / ca < hi]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--index", default=f"{ROOT}/data/index.csv"); args = ap.parse_args()
    dev = "cuda"
    bf = load_birefnet(V2, dev); b5 = load_b5(dev)
    rows = [r for r in csv.DictReader(open(args.index)) if r["split"] == "test"][:args.n]

    METH = ["1024sq", "1536sq", "bucket5"]
    cnt = {s[0]: {"GT": 0, **{m: 0 for m in METH}} for s in SIZES}
    for i, r in enumerate(rows):
        img = np.array(Image.open(r["image"]).convert("RGB")); H, W = img.shape[:2]
        rgb = np.array(Image.open(r["mask"]).convert("RGB")); union = rgb.sum(2) > 0
        ca = float(union.sum())
        if ca < 1500: continue
        allholes = binary_fill_holes(union) & ~union
        x, y, bw, bh = int(r["x"]), int(r["y"]), int(r["bw"]), int(r["bh"])
        box = (x, y, x + bw, y + bh); asp = bw / bh
        sil = {}
        sil["1024sq"] = keep_main(birefnet_outline(bf, img, box, dev, size=1024), 9)
        sil["1536sq"] = keep_main(birefnet_outline(bf, img, box, dev, size=1536), 9)
        sil["bucket5"] = keep_main(outline_bucketed(b5, img, box, route(asp), dev), 9)
        po = {m: (binary_fill_holes(sil[m]) & ~sil[m]) for m in METH}
        for name, lo, hi in SIZES:
            for comp, area in comps(allholes, ca, lo, hi):
                cnt[name]["GT"] += 1
                for m in METH:
                    if (comp & po[m]).sum() >= 0.3 * area: cnt[name][m] += 1
        if (i + 1) % 50 == 0: print(f"  {i+1}/{len(rows)}", flush=True)

    print("\nStructural punchout-hole recall by size (silhouette fill, vs GT):")
    print(f"  {'size':<7}{'GT#':>6}{'1024sq':>9}{'1536sq':>9}{'bucket5':>9}")
    tot = {"GT": 0, **{m: 0 for m in METH}}
    for name, _, _ in SIZES:
        g = cnt[name]["GT"]; tot["GT"] += g
        for m in METH: tot[m] += cnt[name][m]
        f = lambda m: f"{100*cnt[name][m]/g:.0f}%" if g else "-"
        print(f"  {name:<7}{g:>6}{f('1024sq'):>9}{f('1536sq'):>9}{f('bucket5'):>9}")
    g = max(1, tot["GT"])
    print(f"  {'TOTAL':<7}{tot['GT']:>6}" + "".join(f"{100*tot[m]/g:>8.0f}%" for m in METH))
    print("\n  Q: does bucket5 silhouette match/beat 1536sq on small holes? -> if yes, DROP the 1536 pass.")


if __name__ == "__main__":
    main()
