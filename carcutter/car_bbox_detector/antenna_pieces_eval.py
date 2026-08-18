#!/usr/bin/env python3
"""
Does the antenna model bolt SPURIOUS DISCONNECTED STICKS onto the silhouette?

The A/B eval measured coverage and fragmentation ALONG the mast; it said nothing about extra
components. This one anatomises every piece the antenna model adds to the BiRefNet outline.

For each arm (BiRefNet antenna head, zoom UNet), per image:
  extra = arm_mask & ~outline           -> the pixels the antenna model contributes
  split into connected components >= MINPX, and classify each:
    TP        overlaps GT antenna (white) by >= 10px      -> a real antenna piece
    FP        does not                                    -> a spurious stick
    attached  8-touches the raw outline                   -> hangs off the car
    floating  does not touch it                           -> a detached blob in mid-air
  then apply the deployed cleanup keep_main(outline | arm, close=9) and re-check which
  pieces SURVIVE — floating blobs die unless the 9px close bridges them to the body.

Also reports antenna pixels LOST to keep_main (a real mast tip that got dropped for being detached
is the mirror-image failure of a spurious stick surviving).

Run: PYTHONPATH=.:carcutter/car_bbox_detector/birefnet/BiRefNet \
     python carcutter/car_bbox_detector/antenna_pieces_eval.py [--n 250] [--viz]
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

import cv2
import numpy as np
import rfdetr
import segmentation_models_pytorch as smp
import torch
from PIL import Image

sys.path.insert(0, "carcutter/car_bbox_detector/birefnet/BiRefNet")
from carcutter.car_bbox_detector.antenna_fp_clean import keep_main
from carcutter.car_bbox_detector.build_spliced_eval import PROD5, TRI, Spliced, infer, load_bn
from carcutter.car_bbox_detector.cascade_eval import predict_union
from carcutter.car_bbox_detector.gen_coarse_masks import coarse_prob
from carcutter.car_bbox_detector.probe_birefnet_aspect import route

ROOT = "carcutter/car_bbox_detector"
EXP = f"{ROOT}/experiments"
CARDET = f"{EXP}/rfdetr_ca_unified_v1/checkpoint_best_ema.pth"
WHITE = (255, 255, 255)
ANT_CLS = 1
MINPX = 25
TP_OVERLAP = 10
dev = "cuda"


def boxmask(box, H, W):
    m = np.zeros((H, W), np.uint8)
    x0, y0, x1, y1 = [int(v) for v in box]
    m[max(0, y0):min(H, y1 + 1), max(0, x0):min(W, x1 + 1)] = 255
    return m


def pieces(extra, outline, gt_a, kept):
    """Classify each added component: (is_tp, is_attached, survived, area, mask)."""
    n, lbl, st, _ = cv2.connectedComponentsWithStats(extra.astype(np.uint8), 8)
    out = []
    near = cv2.dilate(outline.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    for i in range(1, n):
        a = st[i, cv2.CC_STAT_AREA]
        if a < MINPX:
            continue
        m = lbl == i
        out.append(dict(tp=(m & gt_a).sum() >= TP_OVERLAP,
                        attached=bool((m & near).any()),
                        survived=bool((m & kept).sum() >= 0.5 * a),
                        area=int(a), mask=m))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=f"{EXP}/unet_antenna_v3/checkpoints/best.pt")
    ap.add_argument("--index", default=f"{ROOT}/data/index.csv")
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--det-thr", type=float, default=0.5)
    ap.add_argument("--viz", action="store_true", help="render the worst spurious-stick offenders")
    ap.add_argument("--viz-out", default=f"{ROOT}/data/antenna_sticks.png")
    args = ap.parse_args()

    spliced = Spliced(load_bn(PROD5, 1, dev), load_bn(TRI, 3, dev)).eval()
    cardet = rfdetr.RFDETRMedium.from_checkpoint(CARDET)
    unet = smp.Unet("efficientnet-b4", encoder_weights=None, in_channels=3, classes=1).to(dev).eval()
    st = torch.load(args.ckpt, map_location=dev)
    unet.load_state_dict(st.get("model", st))

    rows = [r for r in csv.DictReader(open(args.index)) if r["split"] == "test"]
    rows = [r for r in rows if r["has_antenna"] == "1"][:args.n]
    print(f"antenna-present test images: {len(rows)}   ckpt: {args.ckpt}")

    S = {a: defaultdict(int) for a in ("head", "unet")}
    lost = defaultdict(list)
    percc = {a: [] for a in ("head", "unet")}
    fpdist = {a: [] for a in ("head", "unet")}   # (px to nearest GT antenna, area) per surviving FP
    worst = []

    for i, r in enumerate(rows):
        pil = Image.open(r["image"]).convert("RGB")
        img = np.array(pil)
        H, W = img.shape[:2]
        rgb = np.array(Image.open(r["mask"]).convert("RGB"))
        gt_o, gt_a = rgb.sum(2) > 0, (rgb == WHITE).all(2)
        if gt_o.sum() < 1500 or gt_a.sum() < 20:
            continue
        carbox = predict_union(cardet, pil, 0.3, 0.15)
        if carbox is None:
            continue
        cx0, cy0, cx1, cy1 = [int(v) for v in carbox]
        cx0, cy0 = max(0, cx0), max(0, cy0)
        cx1, cy1 = min(W, cx1), min(H, cy1)
        if cx1 - cx0 < 32 or cy1 - cy0 < 32:
            continue

        o, _t, head = infer(spliced, img, (cx0, cy0, cx1, cy1), route((cx1 - cx0) / (cy1 - cy0)), dev)
        dd = cardet.predict(Image.fromarray(img[cy0:cy1, cx0:cx1]), threshold=args.det_thr)
        cls = np.asarray(dd.class_id) if getattr(dd, "class_id", None) is not None else np.zeros(len(dd.xyxy))
        um = np.zeros((H, W), bool)
        for bi, b in enumerate(np.asarray(dd.xyxy)):
            if int(cls[bi]) != ANT_CLS:
                continue
            bx0, by0, bx1, by1 = [int(v) for v in b]
            um |= coarse_prob(unet, img, boxmask((cx0 + bx0, cy0 + by0, cx0 + bx1, cy0 + by1), H, W),
                              256, dev) > 0.5

        base = keep_main(o, 9)
        # distance transform of "not GT antenna" -> px from any point to the nearest GT antenna pixel
        d2ant = cv2.distanceTransform((~gt_a).astype(np.uint8), cv2.DIST_L2, 3)
        for arm, m in (("head", head), ("unet", um)):
            kept = keep_main(o | m, 9)
            ps = pieces(m & ~base, base, gt_a, kept)
            percc[arm].append(len(ps))
            for p in ps:
                S[arm]["cc"] += 1
                S[arm]["tp" if p["tp"] else "fp"] += 1
                S[arm]["attached" if p["attached"] else "floating"] += 1
                if p["survived"]:
                    S[arm]["surv"] += 1
                    S[arm]["surv_fp" if not p["tp"] else "surv_tp"] += 1
                    if not p["tp"]:
                        fpdist[arm].append((float(d2ant[p["mask"]].min()), p["area"]))
            S[arm]["imgs"] += 1
            if any(not p["tp"] and p["survived"] for p in ps):
                S[arm]["img_with_surv_fp"] += 1
            # real antenna pixels the cleanup threw away
            arm_a = (m | base) & gt_a
            lost[arm].append((arm_a.sum() - (kept & gt_a).sum()) / max(1, gt_a.sum()))
            if arm == "unet":
                nfp = sum(1 for p in ps if not p["tp"] and p["survived"])
                if nfp:
                    worst.append((nfp, r, base, um, gt_a, ps))

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)

    print(f"\n=== ADDED-PIECE ANATOMY (antenna-present, n={S['unet']['imgs']}, MINPX={MINPX}) ===")
    print(f"  {'arm':<6}{'pieces/img':>11}{'TP':>7}{'FP':>7}{'floating':>10}{'FPsurvive':>11}{'imgs w/ FP stick':>18}{'GT px lost':>12}")
    for a in ("head", "unet"):
        d = S[a]; g = max(1, d["imgs"]); c = max(1, d["cc"])
        print(f"  {a:<6}{np.mean(percc[a]):>11.2f}{100*d['tp']/c:>6.0f}%{100*d['fp']/c:>6.0f}%"
              f"{100*d['floating']/c:>9.0f}%{d['surv_fp']:>10}{100*d['img_with_surv_fp']/g:>17.0f}%"
              f"{100*np.mean(lost[a]):>11.1f}%")
    print("\n  pieces/img = extra components the arm adds to the outline (>=25px)")
    print("  FPsurvive  = spurious sticks still there AFTER keep_main(close=9) — the ones QC would see")
    print("  GT px lost = real antenna pixels keep_main discarded as detached (the opposite failure)")

    for a in ("head", "unet"):
        h = np.bincount(np.array(percc[a], int), minlength=4)[:4]
        print(f"  {a:<6} images by piece count: 0={h[0]} 1={h[1]} 2={h[2]} 3+={sum(percc[a][j] >= 3 for j in range(len(percc[a])))}")

    # Is a surviving "FP" a stray stick elsewhere on the car, or boundary slop hugging the real antenna?
    print(f"\n  surviving FP pieces — distance to the nearest GT antenna pixel")
    print(f"  {'arm':<6}{'n':>4}{'<=10px':>9}{'11-50px':>9}{'>50px':>8}{'med dist':>10}{'med area':>10}")
    for a in ("head", "unet"):
        d = fpdist[a]
        if not d:
            continue
        dd = np.array([x[0] for x in d]); ar = np.array([x[1] for x in d])
        print(f"  {a:<6}{len(d):>4}{(dd <= 10).sum():>9}{((dd > 10) & (dd <= 50)).sum():>9}{(dd > 50).sum():>8}"
              f"{np.median(dd):>10.0f}{np.median(ar):>10.0f}")
    print("  <=10px = boundary slop on the real antenna (cosmetic); >50px = a genuine stray stick")

    if args.viz and worst:
        # Zoom on the spurious component itself — at whole-car scale a 25-500px stick is invisible.
        worst.sort(key=lambda t: -t[0])
        tiles = []
        for nfp, r, base, um, gt_a, ps in worst[:8]:
            img = np.array(Image.open(r["image"]).convert("RGB"))
            fp = max((p for p in ps if not p["tp"] and p["survived"]), key=lambda p: p["area"])
            ys, xs = np.where(fp["mask"])
            cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
            half = max(140, int(2.5 * max(ys.max() - ys.min(), xs.max() - xs.min())))
            y0, y1 = max(0, cy - half), min(img.shape[0], cy + half)
            x0, x1 = max(0, cx - half), min(img.shape[1], cx + half)
            viz = img.copy()
            edge = cv2.dilate(base.astype(np.uint8), np.ones((3, 3), np.uint8)) - base.astype(np.uint8)
            viz[edge > 0] = (0, 200, 0)                                   # silhouette boundary only
            viz[cv2.dilate(gt_a.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0] = (255, 255, 0)
            viz[fp["mask"]] = (255, 0, 0)                                 # the spurious stick
            t = cv2.resize(viz[y0:y1, x0:x1], (640, 640))
            cv2.putText(t, f"{fp['area']}px {'attached' if fp['attached'] else 'FLOATING'}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            tiles.append(t)
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.concatenate([np.concatenate(tiles[i:i + 4], 1) for i in range(0, len(tiles), 4)], 0)
        Image.fromarray(grid).save(args.viz_out)
        print(f"\n  worst spurious-stick cases -> {args.viz_out} (yellow=GT antenna, green=kept real, red=spurious)")


if __name__ == "__main__":
    main()
