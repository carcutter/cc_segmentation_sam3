#!/usr/bin/env python3
"""
Does the zoom-in antenna UNet earn its place on top of the vanilla 5-bucket BiRefNet?

Production framing throughout (no GT boxes): rfdetr_ca_unified car box on the raw frame -> crop ->
aspect-routed spliced BiRefNet (its ch0 outline is byte-identical to birefnet_aspect_prod_bucket5),
and the SAME detector re-run on the car crop for antenna proposals.

Three arms, each a full outline the pipeline could ship:
  A  outline            keep_main(o)                 <- vanilla 5-bucket BiRefNet, the baseline to beat
  B  outline | head     keep_main(o | ch2)           <- free: the antenna head already in the same pass
  C  outline | unet     keep_main(o | zoomUNet)      <- costs a detector pass + a B4 UNet per box

Reported on antenna-present test images:
  cov        recall of GT antenna (white) pixels          <- the headline number
  BF@1       outline boundary-F vs GT union              <- regression guard: adding antenna must not hurt
  FPout      predicted px outside the GT car / pred px   <- what the arm hallucinates
  TIP/MID/BASE + continuity, on thin masts only          <- where a dedicated antenna model should pay off
and on antenna-ABSENT images:
  fire       % of clean cars where the arm bolts something onto the silhouette

Run: PYTHONPATH=.:carcutter/car_bbox_detector/birefnet/BiRefNet \
     python carcutter/car_bbox_detector/antenna_ab_eval.py --ckpt <best.pt> [--n 400]
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
from carcutter.car_bbox_detector.antenna_unet_length_eval import continuity, thin, thirds_cov
from carcutter.car_bbox_detector.build_spliced_eval import PROD5, TRI, Spliced, infer, load_bn
from carcutter.car_bbox_detector.cascade_eval import predict_union
from carcutter.car_bbox_detector.gen_coarse_masks import coarse_prob
from carcutter.car_bbox_detector.probe_birefnet_aspect import route
from carcutter.car_bbox_detector.seg_eval import boundary_f

ROOT = "carcutter/car_bbox_detector"
EXP = f"{ROOT}/experiments"
CARDET = f"{EXP}/rfdetr_ca_unified_v1/checkpoint_best_ema.pth"
WHITE = (255, 255, 255)
ANT_CLS = 1          # rfdetr_ca_unified: 0-indexed 0=car, 1=antenna
MINPX = 25
dev = "cuda"
ARMS = ["outline", "outline|head", "outline|unet"]


def boxmask(box, H, W):
    m = np.zeros((H, W), np.uint8)
    x0, y0, x1, y1 = [int(v) for v in box]
    m[max(0, y0):min(H, y1 + 1), max(0, x0):min(W, x1 + 1)] = 255
    return m


def added_components(base, arm):
    """Components the arm bolts on beyond the vanilla outline, ignoring specks."""
    extra = arm & ~base
    n, _lbl, st, _ = cv2.connectedComponentsWithStats(extra.astype(np.uint8), 8)
    return sum(1 for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= MINPX)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=f"{EXP}/unet_antenna_v3/checkpoints/best.pt")
    ap.add_argument("--index", default=f"{ROOT}/data/index.csv")
    ap.add_argument("--n", type=int, default=0, help="limit antenna-present images (0 = all)")
    ap.add_argument("--n-clean", type=int, default=200, help="antenna-absent images for the fire-rate check")
    ap.add_argument("--det-thr", type=float, default=0.5, help="antenna proposal threshold on the car crop")
    ap.add_argument("--ant-thr", type=float, default=0.5, help="zoom-UNet mask threshold")
    args = ap.parse_args()

    spliced = Spliced(load_bn(PROD5, 1, dev), load_bn(TRI, 3, dev)).eval()
    cardet = rfdetr.RFDETRMedium.from_checkpoint(CARDET)
    unet = smp.Unet("efficientnet-b4", encoder_weights=None, in_channels=3, classes=1).to(dev).eval()
    st = torch.load(args.ckpt, map_location=dev)
    unet.load_state_dict(st.get("model", st))
    print(f"zoom UNet: {args.ckpt}")

    rows = [r for r in csv.DictReader(open(args.index)) if r["split"] == "test"]
    pos = [r for r in rows if r["has_antenna"] == "1"]
    neg = [r for r in rows if r["has_antenna"] == "0"][:args.n_clean]
    if args.n:
        pos = pos[:args.n]
    print(f"antenna-present: {len(pos)}   antenna-absent: {len(neg)}")

    cov = defaultdict(list); bf = defaultdict(list); fpo = defaultdict(list)
    thirds = {a: {"tip": [], "mid": [], "base": [], "cls": defaultdict(int)} for a in ARMS}
    fire = defaultdict(int)
    nthin = 0

    for phase, rowset in (("pos", pos), ("neg", neg)):
        for i, r in enumerate(rowset):
            pil = Image.open(r["image"]).convert("RGB")
            img = np.array(pil)
            H, W = img.shape[:2]
            rgb = np.array(Image.open(r["mask"]).convert("RGB"))
            gt_o = rgb.sum(2) > 0
            gt_a = (rgb == WHITE).all(2)
            if gt_o.sum() < 1500:
                continue

            carbox = predict_union(cardet, pil, 0.3, 0.15)
            if carbox is None:
                continue
            cx0, cy0, cx1, cy1 = [int(v) for v in carbox]
            cx0, cy0 = max(0, cx0), max(0, cy0)
            cx1, cy1 = min(W, cx1), min(H, cy1)
            if cx1 - cx0 < 32 or cy1 - cy0 < 32:
                continue

            o, _tint, head = infer(spliced, img, (cx0, cy0, cx1, cy1), route((cx1 - cx0) / (cy1 - cy0)), dev)

            # antenna proposals from the same detector, run on the car crop (recall needs the crop)
            dd = cardet.predict(Image.fromarray(img[cy0:cy1, cx0:cx1]), threshold=args.det_thr)
            cls = np.asarray(dd.class_id) if getattr(dd, "class_id", None) is not None else np.zeros(len(dd.xyxy))
            unet_mask = np.zeros((H, W), bool)
            for bi, b in enumerate(np.asarray(dd.xyxy)):
                if int(cls[bi]) != ANT_CLS:
                    continue
                bx0, by0, bx1, by1 = [int(v) for v in b]
                gbox = (cx0 + bx0, cy0 + by0, cx0 + bx1, cy0 + by1)
                unet_mask |= coarse_prob(unet, img, boxmask(gbox, H, W), 256, dev) > args.ant_thr

            arms = {"outline": keep_main(o, 9),
                    "outline|head": keep_main(o | head, 9),
                    "outline|unet": keep_main(o | unet_mask, 9)}

            if phase == "neg":
                base = arms["outline"]
                for a in ARMS:
                    if added_components(base, arms[a]):
                        fire[a] += 1
                continue

            if gt_a.sum() < 20:
                continue
            ist, ys, xs = thin(gt_a)
            if ist:
                nthin += 1
                L = max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1)
                tol = max(2, int(round(0.03 * L)))
            for a in ARMS:
                p = arms[a]
                cov[a].append((gt_a & p).sum() / gt_a.sum())
                bf[a].append(boundary_f(gt_o, p, 1))
                fpo[a].append((p & ~gt_o).sum() / max(1, p.sum()))
                if ist:
                    tp, md, bs = thirds_cov(gt_a, p, ys, tol)
                    thirds[a]["tip"].append(tp); thirds[a]["mid"].append(md); thirds[a]["base"].append(bs)
                    thirds[a]["cls"][continuity(gt_a, p, ys, max(1, int(round(0.02 * L))))] += 1

            if (i + 1) % 100 == 0:
                print(f"  {phase} {i+1}/{len(rowset)}", flush=True)

    npos = len(cov[ARMS[0]])
    print(f"\n=== ANTENNA A/B — production framing (n={npos} antenna images, {nthin} thin masts) ===")
    print(f"  {'arm':<14}{'cov':>7}{'BF@1':>8}{'FPout':>8}")
    for a in ARMS:
        print(f"  {a:<14}{np.mean(cov[a]):>7.3f}{np.mean(bf[a]):>8.3f}{np.mean(fpo[a]):>8.3f}")

    print(f"\n  thin masts (n={nthin}) — length coverage along the mast")
    print(f"  {'arm':<14}{'TIP':>6}{'MID':>6}{'BASE':>6}   {'clean':>7}{'frag':>6}{'tiploss':>9}{'miss':>6}")
    for a in ARMS:
        d = thirds[a]; c = d["cls"]; g = max(1, nthin)
        print(f"  {a:<14}{np.nanmean(d['tip'])*100:>5.0f}%{np.nanmean(d['mid'])*100:>5.0f}%{np.nanmean(d['base'])*100:>5.0f}%"
              f"   {100*c['clean']/g:>6.0f}%{100*c['frag']/g:>5.0f}%{100*c['tiploss']/g:>8.0f}%{100*c['miss']/g:>5.0f}%")

    print(f"\n  false fire on antenna-ABSENT cars (n={len(neg)}) — adds a >={MINPX}px blob to the silhouette")
    for a in ARMS:
        print(f"  {a:<14}{100*fire[a]/max(1,len(neg)):>5.0f}%")

    print("\n  refs: v1 zoom UNet on GT boxes tip95/mid95/base96 | spliced head tip71/mid66/base46")
    print("        eval_e2e_prod v1 antenna cov .678 | outline already owns most antenna (cov .78 vs prod .65)")
    print("  DECISION: ship the UNet only if cov/thin-mast gain over 'outline' beats its FPout + fire cost.")


if __name__ == "__main__":
    main()
