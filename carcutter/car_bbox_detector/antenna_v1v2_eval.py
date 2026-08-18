#!/usr/bin/env python3
"""
Antenna v1 vs v2 (hard-negative retrained) — deploy-path eval focused on FALSE POSITIVES.

For each test image, assemble the antenna mask the deploy way: detector antenna boxes
-> coarse_prob(256) per box -> colfill -> union. Compares v1 vs v2 on:
  - white recall / IoU (must not regress)               [antenna-present images]
  - FP-pixel rate = pred pixels outside dilated GT white / pred pixels
  - FP components (pred components with <15% overlap w/ GT white)
  - FP on NO-antenna cars: % of no-antenna images where the model fires (any pred)
The last two target the user-flagged stick-like FP blobs.

Run from repo root:
  PYTHONPATH=. python carcutter/car_bbox_detector/antenna_v1v2_eval.py --n 300
"""
import argparse, csv
import numpy as np, cv2, torch
from PIL import Image
import segmentation_models_pytorch as smp
import rfdetr
from carcutter.car_bbox_detector.gen_coarse_masks import coarse_prob
from carcutter.car_bbox_detector.antenna_continuity import colfill
from carcutter.car_bbox_detector.seg_eval import mask_iou
WHITE = (255, 255, 255); ANT = 1
EXP = "carcutter/car_bbox_detector/experiments"


def load(ck, dev):
    m = smp.Unet("efficientnet-b4", encoder_weights=None, in_channels=3, classes=1).to(dev).eval()
    st = torch.load(ck, map_location=dev); m.load_state_dict(st.get("model", st)); return m


def boxmask(b, H, W):
    m = np.zeros((H, W), np.uint8); x0, y0, x1, y1 = [int(v) for v in b]
    m[max(0, y0):min(H, y1 + 1), max(0, x0):min(W, x1 + 1)] = 255; return m


def assemble(model, img, boxes, dev):
    H, W = img.shape[:2]; out = np.zeros((H, W), bool)
    for b in boxes:
        out |= colfill(coarse_prob(model, img, boxmask(b, H, W), 256, dev) > 0.5)
    return out


def ncomp_fp(pred, white):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    wd = cv2.dilate(white.astype(np.uint8), k) > 0
    n, lbl = cv2.connectedComponents(pred.astype(np.uint8)); fp = 0
    for i in range(1, n):
        comp = lbl == i
        if (comp & wd).sum() < 0.15 * comp.sum():
            fp += 1
    return fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="carcutter/car_bbox_detector/data/index.csv")
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()
    dev = "cuda"
    det = rfdetr.RFDETRMedium.from_checkpoint(f"{EXP}/rfdetr_car_antenna_v1/checkpoint_best_total.pth")
    v1 = load(f"{EXP}/unet_antenna_v1/checkpoints/best.pt", dev)
    v2 = load(f"{EXP}/unet_antenna_v2/checkpoints/best.pt", dev)
    rows = [r for r in csv.DictReader(open(args.index)) if r["split"] == "test"][:args.n]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    agg = {m: {"rec": [], "iou": [], "fprate": [], "fpcc": [], "fire_noant": []} for m in ("v1", "v2")}
    for i, r in enumerate(rows):
        pil = Image.open(r["image"]).convert("RGB"); img = np.array(pil); H, W = img.shape[:2]
        white = (np.array(Image.open(r["mask"]).convert("RGB")) == WHITE).all(2)
        d = det.predict(pil, threshold=0.15)
        cls, conf, xyxy = np.asarray(d.class_id), np.asarray(d.confidence), np.asarray(d.xyxy)
        boxes = [xyxy[j] for j in range(len(xyxy)) if cls[j] == ANT and conf[j] >= 0.15]
        has_ant = white.sum() >= 20
        wd = cv2.dilate(white.astype(np.uint8), k) > 0
        for name, model in (("v1", v1), ("v2", v2)):
            pred = assemble(model, img, boxes, dev)
            if has_ant:
                agg[name]["rec"].append((white & pred).sum() / white.sum())
                agg[name]["iou"].append(mask_iou(white, pred))
                if pred.sum() > 0:
                    agg[name]["fprate"].append((pred & ~wd).sum() / pred.sum())
                agg[name]["fpcc"].append(ncomp_fp(pred, white))
            else:
                agg[name]["fire_noant"].append(1.0 if pred.sum() >= 20 else 0.0)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)

    m = lambda a: float(np.mean(a)) if a else float("nan")
    print(f"\nAntenna v1 vs v2 (deploy path), n={len(rows)}:\n")
    print(f"  {'model':<5}{'recall':>9}{'IoU':>9}{'FPrate':>9}{'FP_cc':>9}{'fire_noAnt%':>13}")
    for name in ("v1", "v2"):
        a = agg[name]
        print(f"  {name:<5}{m(a['rec']):>9.3f}{m(a['iou']):>9.3f}{m(a['fprate']):>9.3f}{m(a['fpcc']):>9.2f}{100*m(a['fire_noant']):>12.1f}%")
    print(f"\n(recall/IoU/FPrate/FP_cc on {len(agg['v1']['rec'])} antenna imgs; "
          f"fire_noAnt on {len(agg['v1']['fire_noant'])} no-antenna imgs — lower FP/fire better)")


if __name__ == "__main__":
    main()
