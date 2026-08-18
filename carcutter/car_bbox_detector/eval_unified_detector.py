#!/usr/bin/env python3
"""
Acceptance check for the unified {car,antenna} detector vs the 2 separate baselines.
  CAR (on RAW frame): box IoU + recall@0.5 vs GT car-extent box (index x,y,bw,bh).
     unified.car  vs  rfdetr_car_antenna_v1.car
  ANTENNA (on car CROP, pad 0.15): instance recall @overlap 0.1/0.3 vs GT antenna CCs.
     unified.antenna  vs  rfdetr_multiclass_v1.antenna
Unified must not regress either. Class ids (0-indexed predict): unified 0=car,1=antenna;
car_antenna_v1 0=car; multiclass_v1 0=antenna.
Run: PYTHONPATH=. HF_HOME=... PY eval_unified_detector.py --unified experiments/rfdetr_ca_unified_v1
"""
import argparse, csv, os
from collections import defaultdict
import numpy as np
from PIL import Image
import rfdetr

ROOT = "carcutter/car_bbox_detector"; EXP = f"{ROOT}/experiments"; WHITE = (255, 255, 255)
CROP_PAD = 0.15; MINPX = 20; DET = 0.3


def find_ckpt(d):
    for n in ("checkpoint_best_ema.pth", "checkpoint_best_regular.pth", "checkpoint_best_total.pth", "checkpoint.pth"):
        if os.path.exists(f"{d}/{n}"): return f"{d}/{n}"
    raise FileNotFoundError(d)


def iou_box(a, b):
    ax0, ay0, ax1, ay1 = a; bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0); ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1-ix0), max(0, iy1-iy0); inter = iw*ih
    ua = (ax1-ax0)*(ay1-ay0) + (bx1-bx0)*(by1-by0) - inter
    return inter/ua if ua > 0 else 0.0


def car_union(det, pil, cls_id):
    dd = det.predict(pil, threshold=DET)
    cls = np.asarray(dd.class_id) if getattr(dd, "class_id", None) is not None else np.zeros(len(dd.xyxy))
    bx = [b for b, c in zip(np.asarray(dd.xyxy), cls) if int(c) == cls_id]
    if not bx: return None
    bx = np.array(bx); return [bx[:,0].min(), bx[:,1].min(), bx[:,2].max(), bx[:,3].max()]


def ant_boxes(det, crop_pil, cls_id):
    dd = det.predict(crop_pil, threshold=DET)
    cls = np.asarray(dd.class_id) if getattr(dd, "class_id", None) is not None else np.zeros(len(dd.xyxy))
    return [b for b, c in zip(np.asarray(dd.xyxy), cls) if int(c) == cls_id]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unified", default=f"{EXP}/rfdetr_ca_unified_v1")
    ap.add_argument("--car-base", default=f"{EXP}/rfdetr_car_antenna_v1")
    ap.add_argument("--ant-base", default=f"{EXP}/rfdetr_multiclass_v1")
    ap.add_argument("--n", type=int, default=930); args = ap.parse_args()
    uni = rfdetr.RFDETRMedium.from_checkpoint(find_ckpt(args.unified))
    carb = rfdetr.RFDETRMedium.from_checkpoint(find_ckpt(args.car_base))
    antb = rfdetr.RFDETRMedium.from_checkpoint(find_ckpt(args.ant_base))
    rows = [r for r in csv.DictReader(open(f"{ROOT}/data/index.csv")) if r["split"] == "test"][:args.n]

    car = {"uni": [], "base": []}      # IoU lists
    ant = {m: {t: [0,0] for t in (0.1,0.3)} for m in ("uni","base")}  # [hit, total]
    for r in rows:
        pil = Image.open(r["image"]).convert("RGB"); W, H = pil.size
        gtc = [int(r["x"]), int(r["y"]), int(r["x"])+int(r["bw"]), int(r["y"])+int(r["bh"])]
        for nm, det, cid in (("uni", uni, 0), ("base", carb, 0)):
            u = car_union(det, pil, cid); car[nm].append(iou_box(u, gtc) if u else 0.0)
        # antenna on crop
        rgb = np.array(Image.open(r["mask"]).convert("RGB")); white = (rgb == WHITE).all(2)
        if white.sum() < MINPX: continue
        import cv2
        x,y,bw,bh = int(r["x"]),int(r["y"]),int(r["bw"]),int(r["bh"])
        px,py = int(bw*CROP_PAD), int(bh*CROP_PAD)
        cx0,cy0 = max(0,x-px),max(0,y-py); cx1,cy1 = min(W,x+bw+px),min(H,y+bh+py)
        crop = pil.crop((cx0,cy0,cx1,cy1))
        n,lbl,st,_ = cv2.connectedComponentsWithStats(white.astype(np.uint8),8)
        gtinst = [(lbl==i) for i in range(1,n) if st[i,4] >= MINPX]
        for nm, det, cid in (("uni", uni, 1), ("base", antb, 0)):
            boxes = ant_boxes(det, crop, cid)
            bmask = np.zeros((H,W), bool)
            for b in boxes:
                bmask[max(0,cy0+int(b[1])):cy0+int(b[3]), max(0,cx0+int(b[0])):cx0+int(b[2])] = True
            for inst in gtinst:
                ov = (inst & bmask).sum()/inst.sum()
                for t in (0.1,0.3):
                    ant[nm][t][1]+=1; ant[nm][t][0]+= int(ov>=t)

    print(f"\n=== UNIFIED vs SEPARATE detectors (n={len(rows)}) ===")
    print(f"CAR (raw frame) box IoU vs GT extent:")
    for nm,lab in (("uni","unified"),("base","car_antenna_v1")):
        a = np.array(car[nm]); print(f"  {lab:<16} meanIoU {a.mean():.3f}  recall@.5 {100*(a>=.5).mean():.0f}%  recall@.7 {100*(a>=.7).mean():.0f}%")
    print(f"ANTENNA (car crop) instance recall:")
    for nm,lab in (("uni","unified"),("base","multiclass_v1")):
        print(f"  {lab:<16} @0.1 {100*ant[nm][0.1][0]/max(1,ant[nm][0.1][1]):.0f}%  @0.3 {100*ant[nm][0.3][0]/max(1,ant[nm][0.3][1]):.0f}%  (n={ant[nm][0.3][1]})")


if __name__ == "__main__":
    main()
