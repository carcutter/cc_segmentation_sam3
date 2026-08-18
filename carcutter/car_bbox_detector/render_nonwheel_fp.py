#!/usr/bin/env python3
"""
Render the NON-WHEEL small-hole FALSE-POSITIVE cases (the 397 FPs that dominate our punchout-precision
deficit). Runs the production pipeline, finds small predicted-punchout components that (a) are NOT in the
GT wheel region and (b) don't overlap a real GT hole (FP), crops each, overlays it red (+ any nearby GT
holes green), and montages the worst (largest-area) ones so we can see what surfaces it false-fires on.
Out: experiments/spliced_viz/nonwheel_hole_fp.png
"""
import csv, os, sys, heapq
import numpy as np, cv2, torch
from PIL import Image
from scipy.ndimage import binary_fill_holes
import segmentation_models_pytorch as smp
import rfdetr
sys.path.insert(0, "carcutter/car_bbox_detector/birefnet/BiRefNet")
from carcutter.car_bbox_detector.probe_birefnet_aspect import route
from carcutter.car_bbox_detector.build_spliced_eval import Spliced, load_bn, infer, PROD5, TRI
from carcutter.car_bbox_detector.gen_coarse_masks import coarse_prob
from carcutter.car_bbox_detector.cascade_eval import predict_union
from carcutter.car_bbox_detector.antenna_fp_clean import keep_main
from carcutter.car_bbox_detector.eval_e2e_prod import (boxmask, segment_box, wheelmask, comps,
                                                       DET_THR, ANT_THR, HOLE_THR, SEG_REG, ANT_SZ, ANTENNA, HOLE)

ROOT = "carcutter/car_bbox_detector"; EXP = f"{ROOT}/experiments"; OUTD = f"{EXP}/spliced_viz"; WHITE = (255, 255, 255); dev = "cuda"
KEEP = 30


def main():
    spliced = Spliced(load_bn(PROD5, 1, dev), load_bn(TRI, 3, dev)).eval()
    cardet = rfdetr.RFDETRMedium.from_checkpoint(f"{EXP}/rfdetr_car_antenna_v1/checkpoint_best_total.pth")
    unidet = rfdetr.RFDETRMedium.from_checkpoint(f"{EXP}/rfdetr_multiclass_v1/checkpoint_best_ema.pth")
    antunet = smp.Unet("efficientnet-b4", encoder_weights=None, in_channels=3, classes=1).to(dev).eval()
    antunet.load_state_dict((lambda s: s.get("model", s))(torch.load(f"{EXP}/unet_antenna_v1/checkpoints/best.pt", map_location=dev)))
    holeunet = smp.Unet("efficientnet-b2", encoder_weights=None, in_channels=3, classes=1).to(dev).eval()
    holeunet.load_state_dict((lambda s: s.get("model", s))(torch.load(f"{EXP}/unet_holeseg_v2/best.pt", map_location=dev)))
    rows = [r for r in csv.DictReader(open(f"{ROOT}/data/index.csv")) if r["split"] == "test"]

    heap = []  # (area_frac, uid, crop)
    uid = 0
    for i, r in enumerate(rows):
        pil = Image.open(r["image"]).convert("RGB"); img = np.array(pil); H, W = img.shape[:2]
        rgb = np.array(Image.open(r["mask"]).convert("RGB"))
        gt_o = rgb.sum(2) > 0
        if gt_o.sum() < 1500: continue
        ca = float(gt_o.sum())
        carbox = predict_union(cardet, pil, 0.3, 0.15)
        if carbox is None: continue
        cx0, cy0, cx1, cy1 = [max(0, int(carbox[0])), max(0, int(carbox[1])), min(W, int(carbox[2])), min(H, int(carbox[3]))]
        if cx1 - cx0 < 32 or cy1 - cy0 < 32: continue
        o, _t, _a = infer(spliced, img, (cx0, cy0, cx1, cy1), route((cx1 - cx0) / (cy1 - cy0)), dev)
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
        outline = keep_main(o | ant_mask, 9)
        punchout = (binary_fill_holes(outline) & ~outline) | hole2
        gt_h = binary_fill_holes(gt_o) & ~gt_o
        wheel_region = binary_fill_holes(cv2.dilate(wheelmask(rgb).astype(np.uint8), np.ones((5,5),np.uint8)) > 0)
        # non-wheel small FP components
        for comp, area in comps(punchout, ca, 0, 0.005):
            if (comp & wheel_region).sum() >= 0.5 * area: continue       # wheel -> skip
            if (comp & gt_h).sum() >= 0.3 * area: continue               # real hole -> not FP
            ys, xs = np.where(comp); pad = 55
            ry0, ry1 = max(0, ys.min()-pad), min(H, ys.max()+pad); rx0, rx1 = max(0, xs.min()-pad), min(W, xs.max()+pad)
            cr = img[ry0:ry1, rx0:rx1].copy()
            cm = comp[ry0:ry1, rx0:rx1]; gh = gt_h[ry0:ry1, rx0:rx1]
            cr[cm] = (0.4 * cr[cm] + np.array([235, 30, 30])).clip(0,255).astype(np.uint8)   # FP red
            ge = gh ^ (cv2.erode(gh.astype(np.uint8), np.ones((3,3),np.uint8)) > 0)
            cr[ge] = [0, 255, 0]                                                              # real GT holes green (context)
            cr = cv2.resize(cr, (260, 260))
            cv2.putText(cr, f"{1000*area/ca:.1f}/k {os.path.basename(r['image'])[:14]}", (4,18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)
            uid += 1
            if len(heap) < KEEP: heapq.heappush(heap, (area/ca, uid, cr))
            elif area/ca > heap[0][0]: heapq.heappushpop(heap, (area/ca, uid, cr))
        if (i+1) % 150 == 0: print(f"  {i+1}/{len(rows)}  (fp collected so far cap {len(heap)})", flush=True)

    items = sorted(heap, key=lambda z: -z[0])
    tiles = [c for _, _, c in items]
    rows_ = [np.hstack([np.hstack([t, np.full((260,3,3),255,np.uint8)]) for t in tiles[i:i+6]]) for i in range(0, len(tiles), 6)]
    w = max(r.shape[1] for r in rows_); rows_ = [np.hstack([r, np.full((260, w-r.shape[1],3),30,np.uint8)]) if r.shape[1]<w else r for r in rows_]
    grid = np.vstack([np.vstack([r, np.full((3,w,3),255,np.uint8)]) for r in rows_])
    cv2.imwrite(f"{OUTD}/nonwheel_hole_fp.png", cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"\n  {len(tiles)} worst non-wheel hole FPs -> {OUTD}/nonwheel_hole_fp.png (RED=spurious hole, green=real GT holes)")


if __name__ == "__main__":
    main()
