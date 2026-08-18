#!/usr/bin/env python3
"""
Head-to-head eval on the test split: trained RF-DETR vs production vs GT.

Coverage-first metrics (clipping the car is the expensive error):
  IoU, top_clip px, any_clip px, fully-contains-GT %.
Reported for ALL test images and the ANTENNA subset.

Run with the cc_sam3 env:
    /home/rutger/miniconda3/envs/cc_sam3/bin/python eval_car_bbox.py \
        --weights experiments/rfdetr_car_bbox_v1/checkpoint_best_total.pth
"""
import argparse, csv
import numpy as np
from PIL import Image


def iou(a, b):
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def metrics(gt, pred):
    return {
        "iou": iou(gt, pred),
        "top_clip": max(0, pred[1] - gt[1]),
        "any_clip": max(pred[0]-gt[0], pred[1]-gt[1], gt[2]-pred[2], gt[3]-pred[3], 0),
        "contained": int(pred[0] <= gt[0] and pred[1] <= gt[1] and pred[2] >= gt[2] and pred[3] >= gt[3]),
    }


def prod_bbox(path):
    a = np.array(Image.open(path)); b = a > 0 if a.ndim == 2 else a.sum(2) > 0
    ys, xs = np.where(b)
    return None if len(ys) == 0 else [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def summarize(label, recs, key):
    recs = [r for r in recs if r.get(key)]
    if not recs:
        print(f"  {label}: (none)"); return
    g = lambda f: np.array([r[key][f] for r in recs])
    print(f"  {label} (n={len(recs)}): IoU {g('iou').mean():.3f} | "
          f"top_clip mean {g('top_clip').mean():5.1f} p90 {np.percentile(g('top_clip'),90):5.1f} "
          f"max {g('top_clip').max():3.0f} | %clip>2px {100*(g('top_clip')>2).mean():4.1f} | "
          f"contains {100*g('contained').mean():4.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="experiments/rfdetr_car_bbox_v1/checkpoint_best_total.pth")
    ap.add_argument("--size", default="medium")
    ap.add_argument("--index", default="data/index.csv")
    ap.add_argument("--threshold", type=float, default=0.3)
    args = ap.parse_args()

    import rfdetr
    DET = {"nano":"RFDETRNano","small":"RFDETRSmall","medium":"RFDETRMedium","base":"RFDETRBase","large":"RFDETRLarge"}
    model = getattr(rfdetr, DET[args.size]).from_checkpoint(args.weights)

    test = [r for r in csv.DictReader(open(args.index)) if r["split"] == "test"]
    recs = []
    for r in test:
        gx, gy, gw, gh = int(r["x"]), int(r["y"]), int(r["bw"]), int(r["bh"])
        gt = [gx, gy, gx+gw, gy+gh]
        pil = Image.open(r["image"]).convert("RGB")
        det = model.predict(pil, threshold=args.threshold)
        rec = {"antenna": int(r["has_antenna"])}
        if len(det.xyxy):
            i = int(np.argmax(det.confidence))            # highest-conf car box
            rec["model"] = metrics(gt, [float(v) for v in det.xyxy[i]])
        if r["prod_outline"]:
            pb = prod_bbox(r["prod_outline"])
            if pb: rec["prod"] = metrics(gt, pb)
        recs.append(rec)

    miss = sum(1 for r in recs if "model" not in r)
    print(f"Test n={len(recs)} (model produced no box on {miss})  threshold={args.threshold}\n")
    for sub, sel in [("ALL", recs), ("ANTENNA", [r for r in recs if r['antenna']]),
                     ("non-antenna", [r for r in recs if not r['antenna']])]:
        print(sub)
        summarize("  MODEL", sel, "model")
        summarize("  PROD ", sel, "prod")
        print()


if __name__ == "__main__":
    main()
