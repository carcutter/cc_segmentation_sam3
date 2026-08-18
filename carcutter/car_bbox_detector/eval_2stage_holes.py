#!/usr/bin/env python3
"""
Evaluate the 2-stage small-hole pipeline: hole-region DETECTOR (RF-DETR) -> tight crop ->
dedicated hole SEGMENTER (UNet) -> stitch. Union with the body-matte punchout (good on
med/large) and count structural-hole instance recall + precision by size, vs prod.

Run from repo root:
  PYTHONPATH=. python carcutter/car_bbox_detector/eval_2stage_holes.py --det-thr 0.3 --seg-thr 0.5
"""
import argparse, csv, os
from pathlib import Path
import numpy as np, cv2, torch
import segmentation_models_pytorch as smp
import rfdetr
from PIL import Image
from scipy.ndimage import binary_fill_holes
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
EXP = "carcutter/car_bbox_detector/experiments"
SMALL_FRAC = 0.006; MINPX = 25; SZ = 512   # MUST match train_smallhole TileDS sz=512 (trains on upsampled crops)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default=f"{EXP}/rfdetr_hole_v1/checkpoint_best_total.pth")
    ap.add_argument("--seg", default=f"{EXP}/unet_holeseg_v1/best.pt")
    ap.add_argument("--e2e", default="carcutter/car_bbox_detector/data/e2e_v3")
    ap.add_argument("--index", default="carcutter/car_bbox_detector/data/index.csv")
    ap.add_argument("--det-thr", type=float, default=0.3)
    ap.add_argument("--seg-thr", type=float, default=0.5)
    ap.add_argument("--det-class", type=int, default=None, help="if set (e.g. 2=holeregion), keep only that class from a multi-class detector")
    args = ap.parse_args()
    dev = "cuda"
    idx = {Path(r["image"]).stem: r for r in csv.DictReader(open(args.index))}
    det = rfdetr.RFDETRMedium.from_checkpoint(args.det)
    seg = smp.Unet("efficientnet-b2", encoder_weights=None, in_channels=3, classes=1).to(dev).eval()
    seg.load_state_dict(torch.load(args.seg, map_location=dev)["model"])

    # per size: [gt, body-only hit, 2stage-union hit, prod hit]
    cnt = {s[0]: [0, 0, 0, 0] for s in SIZES}
    prec = {"body": [0, 0], "aug": [0, 0]}
    for npz in sorted((Path(args.e2e) / "masks").glob("*.npz")):
        r = idx.get(npz.stem)
        if r is None: continue
        d = np.load(npz)
        if "punchout" not in d: continue
        rgb = np.array(Image.open(r["mask"]).convert("RGB")); union = rgb.sum(2) > 0
        ca = float(union.sum())
        if ca < 1500: continue
        img = np.array(Image.open(r["image"]).convert("RGB")); H, W = img.shape[:2]
        allholes = binary_fill_holes(union) & ~union
        body = d["punchout"].astype(bool)
        # 2-stage: run the hole-region detector on the CAR CROP (deploy car-box), map boxes back
        cb = d.get("carbox")
        if cb is None: continue
        cx0, cy0, cx1, cy1 = [int(v) for v in cb]; cx0, cy0 = max(0, cx0), max(0, cy0); cx1, cy1 = min(W, cx1), min(H, cy1)
        carcrop = img[cy0:cy1, cx0:cx1]
        twostage = np.zeros((H, W), bool)
        if carcrop.shape[0] >= 32 and carcrop.shape[1] >= 32:
            dd = det.predict(Image.fromarray(carcrop), threshold=args.det_thr)
            cls = np.asarray(dd.class_id) if getattr(dd, "class_id", None) is not None else None
            for bi, b in enumerate(np.asarray(dd.xyxy)):
                if args.det_class is not None and cls is not None and int(cls[bi]) != args.det_class:
                    continue
                bx0, by0, bx1, by1 = [int(v) for v in b]
                gbox = (cx0 + bx0, cy0 + by0, cx0 + bx1, cy0 + by1)   # crop coords -> image coords
                pr, bb = segment_box(seg, img, gbox, dev, args.seg_thr)
                if pr is not None:
                    x0, y0, x1, y1 = bb; twostage[y0:y1, x0:x1] |= pr
        aug = body | twostage
        prod = binar(r["prod_outline"].replace("/outline/", "/holes_punchout/")) if (r["prod_outline"] and os.path.exists(r["prod_outline"])) else None
        for name, lo, hi in SIZES:
            for comp, area in comps(allholes, ca, lo, hi):
                cnt[name][0] += 1
                if (comp & body).sum() >= 0.3 * area: cnt[name][1] += 1
                if (comp & aug).sum() >= 0.3 * area: cnt[name][2] += 1
                if prod is not None and (comp & prod).sum() >= 0.3 * area: cnt[name][3] += 1
        for comp, area in comps(body, ca, 0, 1.0):
            prec["body"][1] += 1; prec["body"][0] += int((comp & allholes).sum() >= 0.3 * area)
        for comp, area in comps(aug, ca, 0, 1.0):
            prec["aug"][1] += 1; prec["aug"][0] += int((comp & allholes).sum() >= 0.3 * area)

    print(f"\n2-STAGE structural-hole recall (det_thr={args.det_thr}, seg_thr={args.seg_thr}):")
    print(f"  {'size':<7}{'GT#':>6}{'body-only':>11}{'+2stage':>10}{'prod':>9}")
    tot = [0, 0, 0, 0]
    for name, _, _ in SIZES:
        g, b, a, p = cnt[name]
        for i in range(4): tot[i] += [g, b, a, p][i]
        f = lambda x: f"{100*x/g:.0f}%" if g else "-"
        print(f"  {name:<7}{g:>6}{f(b):>11}{f(a):>10}{f(p):>9}")
    g = tot[0]
    print(f"  {'TOTAL':<7}{g:>6}{100*tot[1]/g:>10.0f}%{100*tot[2]/g:>9.0f}%{100*tot[3]/g:>8.0f}%")
    print(f"  precision: body {100*prec['body'][0]/max(1,prec['body'][1]):.0f}%  +2stage {100*prec['aug'][0]/max(1,prec['aug'][1]):.0f}%")


if __name__ == "__main__":
    main()
