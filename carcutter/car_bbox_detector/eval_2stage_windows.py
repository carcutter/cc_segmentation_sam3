#!/usr/bin/env python3
"""
Phase B eval: 2-stage window see-through pipeline. window-region DETECTOR (RF-DETR) on the
car crop -> tight crop -> dedicated window SEGMENTER (UNet) -> stitch. Target = GT tint
windows (blue). Count instance recall + precision by size vs our existing BiRefNet-tint
baseline (d["holes"]) and vs prod (holes_tint).

Run from repo root (SZ=512 MUST match train_smallhole TileDS sz=512):
  PYTHONPATH=. python carcutter/car_bbox_detector/eval_2stage_windows.py --det-thr 0.3 --seg-thr 0.5
"""
import argparse, csv, os
from pathlib import Path
import numpy as np, cv2, torch
import segmentation_models_pytorch as smp
import rfdetr
from PIL import Image
BLUE = (0, 0, 255)
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
EXP = "carcutter/car_bbox_detector/experiments"
MINPX = 60; SZ = 512   # MUST match train_smallhole TileDS sz=512
SIZES = [("small", 0, 0.005), ("med", 0.005, 0.02), ("large", 0.02, 1.0)]


def binar(p):
    if not p or not os.path.exists(p): return None
    a = np.array(Image.open(p)); return (a.sum(2) > 0) if a.ndim == 3 else (a > 0)


def comps(mask, ca, lo=0, hi=1.0):
    n, lbl, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return [(lbl == i, st[i, cv2.CC_STAT_AREA]) for i in range(1, n)
            if MINPX <= st[i, cv2.CC_STAT_AREA] and lo <= st[i, cv2.CC_STAT_AREA] / ca < hi]


@torch.no_grad()
def segment_box(seg, img, box, dev, thr):
    H, W = img.shape[:2]; x0, y0, x1, y1 = [int(v) for v in box]
    x0, y0 = max(0, x0), max(0, y0); x1, y1 = min(W, x1), min(H, y1)
    if x1 - x0 < 8 or y1 - y0 < 8: return None, None
    tile = cv2.resize(img[y0:y1, x0:x1], (SZ, SZ))
    t = torch.from_numpy(((tile / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]).float().to(dev)
    pr = (torch.sigmoid(seg(t))[0, 0] > thr).cpu().numpy().astype(np.uint8)
    pr = cv2.resize(pr, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST) > 0
    return pr, (x0, y0, x1, y1)


def find_ckpt(d):
    for n in ("checkpoint_best_ema.pth", "checkpoint_best_regular.pth", "checkpoint_best_total.pth"):
        if os.path.exists(f"{d}/{n}"): return f"{d}/{n}"
    raise FileNotFoundError(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default=f"{EXP}/rfdetr_windowcrop_v1")
    ap.add_argument("--seg", default=f"{EXP}/unet_windowseg_v1/best.pt")
    ap.add_argument("--e2e", default="carcutter/car_bbox_detector/data/e2e_v3")
    ap.add_argument("--index", default="carcutter/car_bbox_detector/data/index.csv")
    ap.add_argument("--det-thr", type=float, default=0.3)
    ap.add_argument("--seg-thr", type=float, default=0.5)
    ap.add_argument("--det-class", type=int, default=None, help="if set (e.g. 3=windowregion), keep only that class from a multi-class detector")
    args = ap.parse_args()
    dev = "cuda"
    idx = {Path(r["image"]).stem: r for r in csv.DictReader(open(args.index))}
    det = rfdetr.RFDETRMedium.from_checkpoint(find_ckpt(args.det))
    seg = smp.Unet("efficientnet-b2", encoder_weights=None, in_channels=3, classes=1).to(dev).eval()
    seg.load_state_dict(torch.load(args.seg, map_location=dev)["model"])

    # per size: [gt, existing-birefnet hit, 2stage hit, union hit, prod hit]
    cnt = {s[0]: [0, 0, 0, 0, 0] for s in SIZES}
    prec = {"exist": [0, 0], "2stage": [0, 0], "union": [0, 0]}
    for npz in sorted((Path(args.e2e) / "masks").glob("*.npz")):
        r = idx.get(npz.stem)
        if r is None: continue
        d = np.load(npz)
        if "holes" not in d or "carbox" not in d: continue
        rgb = np.array(Image.open(r["mask"]).convert("RGB")); union = rgb.sum(2) > 0
        ca = float(union.sum())
        if ca < 1500: continue
        gtblue = (rgb == BLUE).all(2)
        if gtblue.sum() < MINPX: continue
        img = np.array(Image.open(r["image"]).convert("RGB")); H, W = img.shape[:2]
        exist = d["holes"].astype(bool)
        # 2-stage: window-region detector on the car crop, map boxes back, segment
        cx0, cy0, cx1, cy1 = [int(v) for v in d["carbox"]]
        cx0, cy0 = max(0, cx0), max(0, cy0); cx1, cy1 = min(W, cx1), min(H, cy1)
        carcrop = img[cy0:cy1, cx0:cx1]
        twostage = np.zeros((H, W), bool)
        if carcrop.shape[0] >= 32 and carcrop.shape[1] >= 32:
            dd = det.predict(Image.fromarray(carcrop), threshold=args.det_thr)
            cls = np.asarray(dd.class_id) if getattr(dd, "class_id", None) is not None else None
            for bi, b in enumerate(np.asarray(dd.xyxy)):
                if args.det_class is not None and cls is not None and int(cls[bi]) != args.det_class:
                    continue
                bx0, by0, bx1, by1 = [int(v) for v in b]
                gbox = (cx0 + bx0, cy0 + by0, cx0 + bx1, cy0 + by1)
                pr, bb = segment_box(seg, img, gbox, dev, args.seg_thr)
                if pr is not None:
                    x0, y0, x1, y1 = bb; twostage[y0:y1, x0:x1] |= pr
        uni = exist | twostage
        prod = binar(r["prod_outline"].replace("/outline/", "/holes_tint/")) if (r["prod_outline"] and os.path.exists(r["prod_outline"])) else None
        for name, lo, hi in SIZES:
            for comp, area in comps(gtblue, ca, lo, hi):
                cnt[name][0] += 1
                if (comp & exist).sum() >= 0.3 * area: cnt[name][1] += 1
                if (comp & twostage).sum() >= 0.3 * area: cnt[name][2] += 1
                if (comp & uni).sum() >= 0.3 * area: cnt[name][3] += 1
                if prod is not None and (comp & prod).sum() >= 0.3 * area: cnt[name][4] += 1
        for comp, area in comps(exist, ca, 0, 1.0):
            prec["exist"][1] += 1; prec["exist"][0] += int((comp & gtblue).sum() >= 0.3 * area)
        for comp, area in comps(twostage, ca, 0, 1.0):
            prec["2stage"][1] += 1; prec["2stage"][0] += int((comp & gtblue).sum() >= 0.3 * area)
        for comp, area in comps(uni, ca, 0, 1.0):
            prec["union"][1] += 1; prec["union"][0] += int((comp & gtblue).sum() >= 0.3 * area)

    print(f"\n2-STAGE window recall (det_thr={args.det_thr}, seg_thr={args.seg_thr}):")
    print(f"  {'size':<7}{'GT#':>6}{'birefnet':>10}{'2stage':>8}{'union':>8}{'prod':>8}")
    tot = [0, 0, 0, 0, 0]
    for name, _, _ in SIZES:
        row = cnt[name]; g = row[0]
        for i in range(5): tot[i] += row[i]
        f = lambda x: f"{100*x/g:.0f}%" if g else "-"
        print(f"  {name:<7}{g:>6}{f(row[1]):>10}{f(row[2]):>8}{f(row[3]):>8}{f(row[4]):>8}")
    g = max(1, tot[0])
    print(f"  {'TOTAL':<7}{tot[0]:>6}{100*tot[1]/g:>9.0f}%{100*tot[2]/g:>7.0f}%{100*tot[3]/g:>7.0f}%{100*tot[4]/g:>7.0f}%")
    for k in ("exist", "2stage", "union"):
        p = prec[k]; print(f"  precision {k:<7}: {100*p[0]/max(1,p[1]):.0f}%  ({p[0]}/{p[1]})")


if __name__ == "__main__":
    main()
