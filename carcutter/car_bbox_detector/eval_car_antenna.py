#!/usr/bin/env python3
"""
Eval the 2-class (car + antenna) detector with the union strategy:
final box = union(best car box, all antenna boxes above --ant-threshold).

Compares MODEL_union vs PROD vs GT-full-extent on the test split, coverage-first,
broken out by antenna subset. Antenna threshold is kept LOW (recall-first — an
extra antenna box can only extend the top, the cheap-error direction).

Run with cc_sam3 env:
    /home/rutger/miniconda3/envs/cc_sam3/bin/python eval_car_antenna.py \
        --weights experiments/rfdetr_car_antenna_v1/checkpoint_best_total.pth
"""
import argparse, csv
import numpy as np
from PIL import Image

CAR, ANTENNA = 0, 1   # class ids (0-indexed as emitted)


def iou(a, b):
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def metrics(gt, p):
    return {"iou": iou(gt, p), "top_clip": max(0, p[1]-gt[1]),
            "any_clip": max(p[0]-gt[0], p[1]-gt[1], gt[2]-p[2], gt[3]-p[3], 0),
            "contained": int(p[0] <= gt[0] and p[1] <= gt[1] and p[2] >= gt[2] and p[3] >= gt[3])}


def union(boxes):
    a = np.array(boxes)
    return [float(a[:,0].min()), float(a[:,1].min()), float(a[:,2].max()), float(a[:,3].max())]


def prod_bbox(path):
    a = np.array(Image.open(path)); b = a > 0 if a.ndim == 2 else a.sum(2) > 0
    ys, xs = np.where(b)
    return None if len(ys) == 0 else [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def summ(label, recs, key):
    recs = [r for r in recs if r.get(key)]
    if not recs:
        print(f"  {label}: (none)"); return
    g = lambda f: np.array([r[key][f] for r in recs])
    print(f"  {label} (n={len(recs)}): IoU {g('iou').mean():.3f} | top_clip mean {g('top_clip').mean():5.1f} "
          f"p90 {np.percentile(g('top_clip'),90):5.1f} max {g('top_clip').max():3.0f} | "
          f"%clip>2px {100*(g('top_clip')>2).mean():4.1f} | contains {100*g('contained').mean():4.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="experiments/rfdetr_car_antenna_v1/checkpoint_best_total.pth")
    ap.add_argument("--size", default="medium")
    ap.add_argument("--index", default="data/index.csv")
    ap.add_argument("--car-threshold", type=float, default=0.3)
    ap.add_argument("--ant-threshold", type=float, default=0.15)  # low: recall-first
    args = ap.parse_args()

    import rfdetr
    DET = {"nano":"RFDETRNano","small":"RFDETRSmall","medium":"RFDETRMedium","base":"RFDETRBase","large":"RFDETRLarge"}
    model = getattr(rfdetr, DET[args.size]).from_checkpoint(args.weights)

    test = [r for r in csv.DictReader(open(args.index)) if r["split"] == "test"]
    recs, no_car, ant_used = [], 0, 0
    for r in test:
        gx, gy, gw, gh = int(r["x"]), int(r["y"]), int(r["bw"]), int(r["bh"])
        gt = [gx, gy, gx+gw, gy+gh]
        det = model.predict(Image.open(r["image"]).convert("RGB"), threshold=args.ant_threshold)
        cls = np.asarray(det.class_id); conf = np.asarray(det.confidence); xyxy = np.asarray(det.xyxy)
        rec = {"antenna": int(r["has_antenna"])}
        car_idx = [i for i in range(len(xyxy)) if cls[i] == CAR and conf[i] >= args.car_threshold]
        if car_idx:
            best = car_idx[int(np.argmax(conf[car_idx]))]
            boxes = [list(xyxy[best])]
            ant_idx = [i for i in range(len(xyxy)) if cls[i] == ANTENNA and conf[i] >= args.ant_threshold]
            if ant_idx:
                ant_used += 1
                boxes += [list(xyxy[i]) for i in ant_idx]
            rec["model"] = metrics(gt, union(boxes))
            rec["car_only"] = metrics(gt, [float(v) for v in xyxy[best]])
        else:
            no_car += 1
        if r["prod_outline"]:
            pb = prod_bbox(r["prod_outline"])
            if pb: rec["prod"] = metrics(gt, pb)
        recs.append(rec)

    print(f"Test n={len(recs)} | no car box on {no_car} | antenna boxes added on {ant_used} | "
          f"car_thr={args.car_threshold} ant_thr={args.ant_threshold}\n")
    for sub, sel in [("ALL", recs), ("ANTENNA", [r for r in recs if r['antenna']]),
                     ("non-antenna", [r for r in recs if not r['antenna']])]:
        print(sub)
        summ("  MODEL(car∪ant)", sel, "model")
        summ("  car-only      ", sel, "car_only")
        summ("  PROD          ", sel, "prod")
        print()


if __name__ == "__main__":
    main()
