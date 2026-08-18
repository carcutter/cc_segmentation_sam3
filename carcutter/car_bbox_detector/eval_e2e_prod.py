#!/usr/bin/env python3
"""
PRODUCTION-FRAMED end-to-end eval, scoped to OUTLINE + PUNCHOUT holes + ANTENNA (tint IGNORED).
Pipeline (real detector framing, with refinements):
  RF-DETR car detector (native img) -> car box -> crop
  SPLICED bucketed BiRefNet (route by car-box aspect) -> outline silhouette `o`
  RF-DETR multiclass on car crop -> antenna boxes (cls0) + hole boxes (cls1)   [window cls2 ignored]
    antenna boxes -> zoom antenna UNet (EffNet-B4 @256, coarse_prob, thr0.5, NO colfill)
    hole boxes    -> punchout hole UNet refiner (EffNet-B2 @512, thr0.6)
  assemble:
    outline  = keep_main(o ∪ antenna)
    punchout = (fill_holes(outline) & ~outline)  ∪  hole-2stage      (structural; bucketed silhouette,
                                                                       NO separate 1536 pass)
    antenna  = antenna UNet mask
Score vs prod: outline BF@1/IoU (prod corrected = prod_outline & ~prod_punchout); punchout instance
recall/precision + FP/FN by size (prod holes_punchout); antenna IoU/coverage/detection (prod = none).
Run: PYTHONPATH=.:.../BiRefNet HF_HOME=... PY eval_e2e_prod.py --n 930
"""
import argparse, csv, os, sys
from collections import defaultdict
import numpy as np, cv2, torch
from PIL import Image
from scipy.ndimage import binary_fill_holes
import segmentation_models_pytorch as smp
import rfdetr
sys.path.insert(0, "carcutter/car_bbox_detector/birefnet/BiRefNet")
from carcutter.car_bbox_detector.seg_eval import boundary_f, mask_iou, MEAN, STD
from carcutter.car_bbox_detector.probe_birefnet_aspect import route
from carcutter.car_bbox_detector.build_spliced_eval import Spliced, load_bn, infer, vg, PROD5, TRI
from carcutter.car_bbox_detector.gen_coarse_masks import coarse_prob
from carcutter.car_bbox_detector.cascade_eval import predict_union
from carcutter.car_bbox_detector.antenna_fp_clean import keep_main

ROOT = "carcutter/car_bbox_detector"; EXP = f"{ROOT}/experiments"; WHITE = (255, 255, 255); dev = "cuda"
WHEEL_COLORS = [(255, 199, 200), (128, 128, 128), (200, 0, 255), (255, 165, 0)]  # FL, BL, BR, FR (data-pipeline legend)
DET_THR = 0.5; ANT_THR = 0.5; HOLE_THR = 0.6; SEG_REG = 512; ANT_SZ = 256
ANTENNA, HOLE = 0, 1
V = ["straight", "corner", "corner34", "side", "other", "ALL"]
SIZES = [("small", 0, 0.005), ("med", 0.005, 0.02), ("large", 0.02, 1.0)]
MINPX = 25


def binar(p):
    if not p or not os.path.exists(p): return None
    a = np.array(Image.open(p)); return (a.sum(2) > 0) if a.ndim == 3 else (a > 0)


def boxmask(box, H, W):
    m = np.zeros((H, W), np.uint8); x0, y0, x1, y1 = [int(v) for v in box]
    m[max(0,y0):min(H,y1+1), max(0,x0):min(W,x1+1)] = 255; return m


@torch.no_grad()
def segment_box(seg, img, box, thr, sz):
    H, W = img.shape[:2]; x0, y0, x1, y1 = [int(v) for v in box]
    x0, y0 = max(0, x0), max(0, y0); x1, y1 = min(W, x1), min(H, y1)
    if x1 - x0 < 8 or y1 - y0 < 8: return None
    tile = cv2.resize(img[y0:y1, x0:x1], (sz, sz))
    t = torch.from_numpy(((tile / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]).float().to(dev)
    pr = (torch.sigmoid(seg(t))[0, 0] > thr).cpu().numpy().astype(np.uint8)
    pr = cv2.resize(pr, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST) > 0
    full = np.zeros((H, W), bool); full[y0:y1, x0:x1] = pr; return full


def comps(mask, ca, lo=0, hi=1.0):
    n, lbl, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return [(lbl == i, st[i, cv2.CC_STAT_AREA]) for i in range(1, n)
            if MINPX <= st[i, cv2.CC_STAT_AREA] and lo <= st[i, cv2.CC_STAT_AREA] / ca < hi]


def hole_stats(gt_h, pred_h, ca, lo=0, hi=1.0, region=None):
    """if region given, count only hole components whose majority lies in `region`."""
    def inreg(comp, area):
        return True if region is None else (comp & region).sum() >= 0.5 * area
    gtc = [(g, a) for g, a in comps(gt_h, ca, lo, hi) if inreg(g, a)]
    prc = [(p, a) for p, a in comps(pred_h, ca, lo, hi) if inreg(p, a)]
    tp = sum(1 for g, a in gtc if (g & pred_h).sum() >= 0.3 * a)
    fp = sum(1 for p, a in prc if (p & gt_h).sum() < 0.3 * a)
    return len(gtc), tp, len(gtc) - tp, fp


def wheelmask(rgb, tol=20):
    m = np.zeros(rgb.shape[:2], bool); rint = rgb.astype(int)
    for c in WHEEL_COLORS:
        m |= np.abs(rint - np.array(c)).max(2) <= tol
    return m


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=930)
    ap.add_argument("--index", default=f"{ROOT}/data/index.csv"); args = ap.parse_args()
    spliced = Spliced(load_bn(PROD5, 1, dev), load_bn(TRI, 3, dev)).eval()
    cardet = rfdetr.RFDETRMedium.from_checkpoint(f"{EXP}/rfdetr_car_antenna_v1/checkpoint_best_total.pth")
    unidet = rfdetr.RFDETRMedium.from_checkpoint(f"{EXP}/rfdetr_multiclass_v1/checkpoint_best_ema.pth")
    antunet = smp.Unet("efficientnet-b4", encoder_weights=None, in_channels=3, classes=1).to(dev).eval()
    antunet.load_state_dict((lambda s: s.get("model", s))(torch.load(f"{EXP}/unet_antenna_v1/checkpoints/best.pt", map_location=dev)))
    holeunet = smp.Unet("efficientnet-b2", encoder_weights=None, in_channels=3, classes=1).to(dev).eval()
    holeunet.load_state_dict((lambda s: s.get("model", s))(torch.load(f"{EXP}/unet_holeseg_v2/best.pt", map_location=dev)))
    rows = [r for r in csv.DictReader(open(args.index)) if r["split"] == "test"][:args.n]

    ob = {k: defaultdict(list) for k in ("ours", "prod")}; oi = {k: defaultdict(list) for k in ("ours", "prod")}
    HOLES = {k: {s[0]: [0,0,0,0] for s in SIZES} for k in ("ours", "prod")}
    HW = {k: {reg: [0,0,0,0] for reg in ("wheel", "nonwheel")} for k in ("ours", "prod")}  # SMALL holes only
    ant = {"iou": [], "cov": [], "det": []}
    nskip = 0
    for i, r in enumerate(rows):
        pil = Image.open(r["image"]).convert("RGB"); img = np.array(pil); H, W = img.shape[:2]
        rgb = np.array(Image.open(r["mask"]).convert("RGB"))
        gt_o = rgb.sum(2) > 0; gt_a = (rgb == WHITE).all(2)
        if gt_o.sum() < 1500: continue
        ca = float(gt_o.sum()); v = vg(os.path.basename(r["image"]))
        carbox = predict_union(cardet, pil, 0.3, 0.15)
        if carbox is None: nskip += 1; continue
        cx0, cy0, cx1, cy1 = [int(x) for x in carbox]
        cx0, cy0 = max(0, cx0), max(0, cy0); cx1, cy1 = min(W, cx1), min(H, cy1)
        if cx1 - cx0 < 32 or cy1 - cy0 < 32: nskip += 1; continue
        # spliced bucketed outline on detector car box
        o, _t, _a = infer(spliced, img, (cx0, cy0, cx1, cy1), route((cx1 - cx0) / (cy1 - cy0)), dev)
        # unified region detector on the car crop
        carcrop = img[cy0:cy1, cx0:cx1]
        dd = unidet.predict(Image.fromarray(carcrop), threshold=DET_THR)
        cls = np.asarray(dd.class_id) if getattr(dd, "class_id", None) is not None else np.zeros(len(dd.xyxy))
        ant_mask = np.zeros((H, W), bool); hole2 = np.zeros((H, W), bool)
        for bi, b in enumerate(np.asarray(dd.xyxy)):
            bx0, by0, bx1, by1 = [int(x) for x in b]; gbox = (cx0 + bx0, cy0 + by0, cx0 + bx1, cy0 + by1)
            c = int(cls[bi])
            if c == ANTENNA:
                ant_mask |= coarse_prob(antunet, img, boxmask(gbox, H, W), ANT_SZ, dev) > ANT_THR
            elif c == HOLE:
                m = segment_box(holeunet, img, gbox, HOLE_THR, SEG_REG)
                if m is not None: hole2 |= m
        # assemble
        outline = keep_main(o | ant_mask, 9)
        punchout = (binary_fill_holes(outline) & ~outline) | hole2
        # --- OUTLINE ---
        prod_o = binar(r["prod_outline"]); prod_p = binar(r["prod_outline"].replace("/outline/", "/holes_punchout/")) if r["prod_outline"] else None
        for vv in (v, "ALL"):
            ob["ours"][vv].append(boundary_f(gt_o, outline, 1)); oi["ours"][vv].append(mask_iou(gt_o, outline))
            if prod_o is not None:
                pc = prod_o & ~prod_p if prod_p is not None else prod_o
                ob["prod"][vv].append(boundary_f(gt_o, pc, 1)); oi["prod"][vv].append(mask_iou(gt_o, pc))
        # --- PUNCHOUT (structural) ---
        gt_h = binary_fill_holes(gt_o) & ~gt_o
        wheel_region = binary_fill_holes(cv2.dilate(wheelmask(rgb).astype(np.uint8), np.ones((5,5),np.uint8)) > 0)
        for nm, pred in (("ours", punchout), ("prod", prod_p)):
            if pred is None: continue
            for s, lo, hi in SIZES:
                for k, val in zip(range(4), hole_stats(gt_h, pred, ca, lo, hi)): HOLES[nm][s][k] += val
            # SMALL holes split wheel vs non-wheel (the precision-deficit class)
            for k, val in zip(range(4), hole_stats(gt_h, pred, ca, 0, 0.005, wheel_region)): HW[nm]["wheel"][k] += val
            for k, val in zip(range(4), hole_stats(gt_h, pred, ca, 0, 0.005, ~wheel_region)): HW[nm]["nonwheel"][k] += val
        # --- ANTENNA (vs GT; prod has none) ---
        if gt_a.sum() >= 20:
            ant["iou"].append(mask_iou(gt_a, ant_mask)); ant["cov"].append((gt_a & ant_mask).sum() / gt_a.sum())
            ant["det"].append(1.0 if (gt_a & ant_mask).sum() >= 0.1 * gt_a.sum() else 0.0)
        if (i + 1) % 100 == 0: print(f"  {i+1}/{len(rows)} (skip {nskip})", flush=True)

    print(f"\n=== PRODUCTION E2E (detector-framed, with refinements; tint ignored). skipped {nskip} (no car box) ===")
    print(f"\nOUTLINE  (prod corrected = prod_outline & ~prod_punchout)")
    print(f"  {'view':<9}{'n':>5}{'OURS_BF1':>10}{'PROD_BF1':>10}{'OURS_IoU':>10}{'PROD_IoU':>10}")
    for v in V:
        if ob["ours"][v]:
            print(f"  {v:<9}{len(ob['ours'][v]):>5}{np.mean(ob['ours'][v]):>10.3f}"
                  f"{(np.mean(ob['prod'][v]) if ob['prod'][v] else float('nan')):>10.3f}"
                  f"{np.mean(oi['ours'][v]):>10.3f}{(np.mean(oi['prod'][v]) if oi['prod'][v] else float('nan')):>10.3f}")
    print(f"\nPUNCHOUT holes (instance, 30% overlap)")
    print(f"  {'':<6}{'size':<7}{'GT#':>6}{'recall':>8}{'prec':>7}{'FP':>6}")
    for nm in ("ours", "prod"):
        tot = [0,0,0,0]
        for s, _, _ in SIZES:
            g, tp, fn, fp = HOLES[nm][s]; [tot.__setitem__(k, tot[k]+[g,tp,fn,fp][k]) for k in range(4)]
            print(f"  {nm:<6}{s:<7}{g:>6}{(100*tp/g if g else 0):>7.0f}%{(100*tp/max(1,tp+fp)):>6.0f}%{fp:>6}")
        g, tp, fn, fp = tot
        print(f"  {nm:<6}{'TOTAL':<7}{g:>6}{(100*tp/max(1,g)):>7.0f}%{(100*tp/max(1,tp+fp)):>6.0f}%{fp:>6}")
    print(f"\nSMALL punchout holes split WHEEL vs NON-WHEEL (wheel region from GT wheel-color labels)")
    print(f"  {'':<6}{'region':<9}{'GT#':>6}{'recall':>8}{'prec':>7}{'FP':>6}")
    for nm in ("ours", "prod"):
        for reg in ("wheel", "nonwheel"):
            g, tp, fn, fp = HW[nm][reg]
            print(f"  {nm:<6}{reg:<9}{g:>6}{(100*tp/g if g else 0):>7.0f}%{(100*tp/max(1,tp+fp)):>6.0f}%{fp:>6}")

    print(f"\nANTENNA (vs GT; prod has no antenna class)  n={len(ant['iou'])}")
    print(f"  IoU {np.mean(ant['iou']):.3f}  coverage {np.mean(ant['cov']):.3f}  detection-rate {np.mean(ant['det']):.3f}")


if __name__ == "__main__":
    main()
