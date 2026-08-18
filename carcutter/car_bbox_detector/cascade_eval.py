#!/usr/bin/env python3
"""
Coarse-to-fine cascade eval (inference-only, uses the existing v2 car+antenna model).

  pass1  = v2(full image)            -> union(car, antenna)            [= eval_v2 baseline]
  crop   = pad(pass1 box): top +TOP_PAD*h, sides/bottom +SIDE_PAD     (clamped to image)
  pass2  = v2(crop), boxes mapped back to full-image coords -> union(car, antenna)
  final  = union(pass1, pass2)       (monotonic — never smaller than pass1)

The generous TOP pad guarantees the antenna stays in the crop even when pass1
clips it; pass2 sees the car filling the frame, so the antenna gets ~2-3x more
effective pixels. Reports PASS1 vs CASCADE vs PROD, coverage-first, antenna subset.

Run: /home/rutger/miniconda3/envs/cc_sam3/bin/python cascade_eval.py \
        --weights experiments/rfdetr_car_antenna_v1/checkpoint_best_total.pth
"""
import argparse, csv
import numpy as np
from PIL import Image

CAR, ANT = 0, 1
TOP_PAD, SIDE_PAD = 0.5, 0.15   # fraction of pass1 box h/w


def iou(a, b):
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1-ix0) * max(0, iy1-iy0)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0


def metrics(gt, p):
    return {"iou": iou(gt, p), "top_clip": max(0, p[1]-gt[1]),
            "any_clip": max(p[0]-gt[0], p[1]-gt[1], gt[2]-p[2], gt[3]-p[3], 0),
            "contained": int(p[0] <= gt[0] and p[1] <= gt[1] and p[2] >= gt[2] and p[3] >= gt[3])}


def union(boxes):
    a = np.array(boxes)
    return [float(a[:,0].min()), float(a[:,1].min()), float(a[:,2].max()), float(a[:,3].max())]


def predict_union(model, pil, car_thr, ant_thr, ox=0.0, oy=0.0):
    """Return union(best car, antennas) in coords offset by (ox,oy), or None if no car."""
    det = model.predict(pil, threshold=ant_thr)
    if not len(det.xyxy):
        return None
    cls, conf, xyxy = np.asarray(det.class_id), np.asarray(det.confidence), np.asarray(det.xyxy)
    car_idx = [i for i in range(len(xyxy)) if cls[i] == CAR and conf[i] >= car_thr]
    if not car_idx:
        return None
    best = car_idx[int(np.argmax(conf[car_idx]))]
    boxes = [list(xyxy[best])]
    boxes += [list(xyxy[i]) for i in range(len(xyxy)) if cls[i] == ANT and conf[i] >= ant_thr]
    u = union(boxes)
    return [u[0]+ox, u[1]+oy, u[2]+ox, u[3]+oy]


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
    ap.add_argument("--ant-threshold", type=float, default=0.15)
    args = ap.parse_args()

    import rfdetr
    DET = {"nano":"RFDETRNano","small":"RFDETRSmall","medium":"RFDETRMedium","base":"RFDETRBase","large":"RFDETRLarge"}
    model = getattr(rfdetr, DET[args.size]).from_checkpoint(args.weights)

    test = [r for r in csv.DictReader(open(args.index)) if r["split"] == "test"]
    recs, no_car, extended = [], 0, 0
    for r in test:
        gx, gy, gw, gh = int(r["x"]), int(r["y"]), int(r["bw"]), int(r["bh"])
        gt = [gx, gy, gx+gw, gy+gh]
        W, H = int(r["width"]), int(r["height"])
        pil = Image.open(r["image"]).convert("RGB")
        rec = {"antenna": int(r["has_antenna"])}

        box1 = predict_union(model, pil, args.car_threshold, args.ant_threshold)
        if box1 is None:
            no_car += 1; recs.append(rec)
            if r["prod_outline"]:
                pb = prod_bbox(r["prod_outline"]);  rec["prod"] = metrics(gt, pb) if pb else None
            continue
        rec["pass1"] = metrics(gt, box1)

        # pad pass1 box, clamp to image, crop, run pass2
        bw_, bh_ = box1[2]-box1[0], box1[3]-box1[1]
        cx0 = max(0, int(box1[0]-SIDE_PAD*bw_)); cy0 = max(0, int(box1[1]-TOP_PAD*bh_))
        cx1 = min(W, int(box1[2]+SIDE_PAD*bw_)); cy1 = min(H, int(box1[3]+SIDE_PAD*bh_))
        crop = pil.crop((cx0, cy0, cx1, cy1))
        box2 = predict_union(model, crop, args.car_threshold, args.ant_threshold, ox=cx0, oy=cy0)
        final = union([box1, box2]) if box2 else box1
        if box2 and (final[1] < box1[1]-0.5 or final[0] < box1[0]-0.5 or final[2] > box1[2]+0.5 or final[3] > box1[3]+0.5):
            extended += 1
        rec["cascade"] = metrics(gt, final)

        if r["prod_outline"]:
            pb = prod_bbox(r["prod_outline"]);  rec["prod"] = metrics(gt, pb) if pb else None
        recs.append(rec)

    print(f"Test n={len(recs)} | no car on {no_car} | cascade extended box on {extended} | "
          f"top_pad={TOP_PAD} side_pad={SIDE_PAD} car_thr={args.car_threshold} ant_thr={args.ant_threshold}\n")
    for sub, sel in [("ALL", recs), ("ANTENNA", [r for r in recs if r['antenna']]),
                     ("non-antenna", [r for r in recs if not r['antenna']])]:
        print(sub)
        summ("  PASS1 (v2)    ", sel, "pass1")
        summ("  CASCADE       ", sel, "cascade")
        summ("  PROD          ", sel, "prod")
        print()


if __name__ == "__main__":
    main()
