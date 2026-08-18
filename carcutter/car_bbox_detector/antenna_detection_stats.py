#!/usr/bin/env python3
"""
Antenna-head detection quality on the test split (full-image / pass-1).

Matches predicted `antenna` boxes (cls==1, conf>=ant_thr) against GT white-class
boxes from coco_ca_test.json, greedily by IoU. Reports instance-level TP/FP/FN
(precision/recall/F1) at a couple IoU thresholds, plus image-level detection
(did we find >=1 antenna on a car that has one?).

Run: /home/rutger/miniconda3/envs/cc_sam3/bin/python antenna_detection_stats.py \
        --weights experiments/rfdetr_car_antenna_v1/checkpoint_best_total.pth
"""
import argparse, json
import numpy as np
from PIL import Image

ANT = 1


def iou(a, b):
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1-ix0) * max(0, iy1-iy0)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0


def match(preds, gts, thr):
    """Greedy IoU match (preds sorted by conf desc). Returns tp, fp, fn."""
    used = [False]*len(gts); tp = 0
    for p in preds:
        best, bi = thr, -1
        for j, g in enumerate(gts):
            if used[j]:
                continue
            v = iou(p, g)
            if v >= best:
                best, bi = v, j
        if bi >= 0:
            used[bi] = True; tp += 1
    fp = len(preds) - tp
    fn = len(gts) - tp
    return tp, fp, fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="experiments/rfdetr_car_antenna_v1/checkpoint_best_total.pth")
    ap.add_argument("--size", default="medium")
    ap.add_argument("--coco", default="data/coco_ca_test.json")
    ap.add_argument("--ant-threshold", type=float, default=0.15)
    args = ap.parse_args()

    coco = json.load(open(args.coco))
    gt_ant = {}                                   # file_name -> list of xyxy GT antenna boxes
    imgs = {im["id"]: im for im in coco["images"]}
    for im in coco["images"]:
        gt_ant[im["file_name"]] = []
    for a in coco["annotations"]:
        if a["category_id"] == 2:
            x, y, w, h = a["bbox"]
            gt_ant[imgs[a["image_id"]]["file_name"]].append([x, y, x+w, y+h])

    import rfdetr
    DET = {"nano":"RFDETRNano","small":"RFDETRSmall","medium":"RFDETRMedium","base":"RFDETRBase","large":"RFDETRLarge"}
    model = getattr(rfdetr, DET[args.size]).from_checkpoint(args.weights)

    THRS = [0.3, 0.5]
    inst = {t: [0, 0, 0] for t in THRS}           # tp,fp,fn
    n_gt_ant = n_pred_ant = 0
    img_with_gt = img_detected = 0                # image-level recall
    img_no_gt = img_false = 0                     # image-level FP (antenna predicted where none)
    confs = []

    for fn_, gts in gt_ant.items():
        det = model.predict(Image.open(fn_).convert("RGB"), threshold=args.ant_threshold)
        cls, conf, xyxy = np.asarray(det.class_id), np.asarray(det.confidence), np.asarray(det.xyxy)
        idx = [i for i in range(len(xyxy)) if cls[i] == ANT and conf[i] >= args.ant_threshold]
        preds = [list(xyxy[i]) for i in sorted(idx, key=lambda i: -conf[i])]
        confs += [float(conf[i]) for i in idx]
        n_gt_ant += len(gts); n_pred_ant += len(preds)
        for t in THRS:
            tp, fp, fn = match(preds, gts, t)
            inst[t][0] += tp; inst[t][1] += fp; inst[t][2] += fn
        if gts:
            img_with_gt += 1; img_detected += int(len(preds) > 0)
        else:
            img_no_gt += 1; img_false += int(len(preds) > 0)

    print(f"Test images: {len(gt_ant)} | GT antenna boxes: {n_gt_ant} | predicted antenna boxes: {n_pred_ant} "
          f"(ant_thr={args.ant_threshold})")
    print(f"pred antenna conf: mean {np.mean(confs):.2f} median {np.median(confs):.2f}\n" if confs else "")
    print("Instance-level (IoU-matched):")
    for t in THRS:
        tp, fp, fn = inst[t]
        prec = tp/(tp+fp) if tp+fp else 0; rec = tp/(tp+fn) if tp+fn else 0
        f1 = 2*prec*rec/(prec+rec) if prec+rec else 0
        print(f"  IoU>={t}: TP {tp:4d}  FP {fp:4d}  FN {fn:4d} | precision {prec:.3f}  recall {rec:.3f}  F1 {f1:.3f}")
    print(f"\nImage-level antenna presence:")
    print(f"  cars WITH antenna: {img_with_gt} | detected >=1: {img_detected} "
          f"({100*img_detected/img_with_gt:.1f}% image-recall)")
    print(f"  cars WITHOUT antenna: {img_no_gt} | falsely predicted >=1: {img_false} "
          f"({100*img_false/max(1,img_no_gt):.1f}% image-FP)")


if __name__ == "__main__":
    main()
