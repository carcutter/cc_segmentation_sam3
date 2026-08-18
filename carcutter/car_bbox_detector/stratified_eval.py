#!/usr/bin/env python3
"""
#16 distribution/gap analysis for the shipping detector (576 v2 + cascade).

Runs the cascade per test image and buckets the coverage-first metrics
(top_clip, %clipped>2px, fully-contains%) by subgroup, to find where we're weak:
  - vehicle-shape proxy: GT bbox aspect ratio w/h (wide=long truck/van; tall=campervan/front view)
  - vehicle-size proxy:  GT area fraction of the image (big vehicles fill the frame)
  - batch / source       (filenames hint at fleet/region/type)
  - antenna presence

Run from repo root:
  PYTHONPATH=. /home/rutger/miniconda3/envs/cc_sam3/bin/python carcutter/car_bbox_detector/stratified_eval.py \
      --weights carcutter/car_bbox_detector/experiments/rfdetr_car_antenna_v1/checkpoint_best_total.pth
"""
import argparse, csv
from collections import defaultdict
import numpy as np
from PIL import Image
from carcutter.car_bbox_detector.cascade_eval import predict_union, union, TOP_PAD, SIDE_PAD


def metrics(gt, p):
    return {"top_clip": max(0, p[1]-gt[1]),
            "contained": int(p[0] <= gt[0] and p[1] <= gt[1] and p[2] >= gt[2] and p[3] >= gt[3])}


def report(name, buckets):
    print(f"\n=== by {name} ===")
    print(f"  {'bucket':<22}{'n':>5}{'top_clip mean':>15}{'%clip>2px':>11}{'contains%':>11}")
    for k in sorted(buckets, key=lambda x: str(x)):
        rs = buckets[k]
        tc = np.array([r["top_clip"] for r in rs]); ct = np.array([r["contained"] for r in rs])
        print(f"  {str(k):<22}{len(rs):>5}{tc.mean():>15.1f}{100*(tc>2).mean():>10.1f}%{100*ct.mean():>10.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="carcutter/car_bbox_detector/experiments/rfdetr_car_antenna_v1/checkpoint_best_total.pth")
    ap.add_argument("--size", default="medium")
    ap.add_argument("--index", default="carcutter/car_bbox_detector/data/index.csv")
    args = ap.parse_args()
    import rfdetr
    model = rfdetr.RFDETRMedium.from_checkpoint(args.weights)

    test = [r for r in csv.DictReader(open(args.index)) if r["split"] == "test"]
    by_ar, by_area, by_batch, by_ant = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    rows = []
    for r in test:
        gx, gy, gw, gh = int(r["x"]), int(r["y"]), int(r["bw"]), int(r["bh"])
        W, H = int(r["width"]), int(r["height"]); gt = [gx, gy, gx+gw, gy+gh]
        pil = Image.open(r["image"]).convert("RGB")
        box1 = predict_union(model, pil, 0.3, 0.15)
        if box1 is None:
            continue
        bw_, bh_ = box1[2]-box1[0], box1[3]-box1[1]
        c = (max(0,int(box1[0]-SIDE_PAD*bw_)), max(0,int(box1[1]-TOP_PAD*bh_)),
             min(W,int(box1[2]+SIDE_PAD*bw_)), min(H,int(box1[3]+SIDE_PAD*bh_)))
        box2 = predict_union(model, pil.crop((c[0],c[1],c[2],c[3])), 0.3, 0.15, ox=c[0], oy=c[1])
        final = union([box1, box2]) if box2 else box1
        m = metrics(gt, final)
        ar = gw/max(1, gh); areaf = (gw*gh)/(W*H)
        arb = "wide >2.2 (long)" if ar > 2.2 else ("tall <1.3 (front/box)" if ar < 1.3 else "normal 1.3-2.2")
        areab = "big >0.55" if areaf > 0.55 else ("small <0.35" if areaf < 0.35 else "mid 0.35-0.55")
        by_ar[arb].append(m); by_area[areab].append(m)
        by_batch[r["batch"].replace("_car_segmentation","")].append(m)
        by_ant["antenna" if int(r["has_antenna"]) else "no-antenna"].append(m)
        rows.append(m)
    tc = np.array([r["top_clip"] for r in rows])
    print(f"OVERALL n={len(rows)}: top_clip mean {tc.mean():.1f} | %clip>2px {100*(tc>2).mean():.1f} | contains {100*np.mean([r['contained'] for r in rows]):.1f}%")
    report("aspect-ratio (vehicle shape)", by_ar)
    report("area-fraction (vehicle size)", by_area)
    report("batch/source", by_batch)
    report("antenna presence", by_ant)


if __name__ == "__main__":
    main()
