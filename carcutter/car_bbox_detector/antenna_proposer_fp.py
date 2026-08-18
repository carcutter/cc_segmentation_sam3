#!/usr/bin/env python3
"""
Proposer FALSE-POSITIVE rate: how many SPURIOUS antenna candidates would each proposer hand the
zoom-in UNet? Per test image, a candidate = an rfdetr antenna box (class0) OR a connected component
of the head mask `a`. TP candidate = overlaps a real GT antenna (>=10px); else FP (spurious).
Key number = the MARGINAL FP cost of ADDING the head to rfdetr = head CCs that don't overlap any
rfdetr box, split TP (real antenna rfdetr missed = the +4% win) vs FP (wild-growth stubs).
Also clean-image fire rate (images with NO GT antenna that still get a candidate).
"""
import csv, os, sys
from collections import defaultdict
import numpy as np, cv2, torch
from PIL import Image
import rfdetr
sys.path.insert(0, "carcutter/car_bbox_detector/birefnet/BiRefNet")
from carcutter.car_bbox_detector.probe_birefnet_aspect import route, crop_pad_box
from carcutter.car_bbox_detector.build_spliced_eval import Spliced, load_bn, infer, PROD5, TRI

ROOT = "carcutter/car_bbox_detector"; EXP = f"{ROOT}/experiments"; WHITE = (255, 255, 255); dev = "cuda"
DET_THR = 0.25; HEAD_MINPX = 15; HIT = 10


def main():
    model = Spliced(load_bn(PROD5, 1, dev), load_bn(TRI, 3, dev)).eval()
    det = rfdetr.RFDETRMedium.from_checkpoint(f"{EXP}/rfdetr_multiclass_v1/checkpoint_best_ema.pth")
    rows = [r for r in csv.DictReader(open(f"{ROOT}/data/index.csv")) if r["split"] == "test"]

    S = defaultdict(int)  # candidate TP/FP tallies + image-level fire-on-clean
    nimg = nclean = 0
    for r in rows:
        rgb = np.array(Image.open(r["mask"]).convert("RGB")); gt_a = (rgb == WHITE).all(2)
        has_ant = gt_a.sum() >= 20
        img = np.array(Image.open(r["image"]).convert("RGB")); H, W = img.shape[:2]
        nimg += 1
        bw, bh = int(r["bw"]), int(r["bh"]); box = (int(r["x"]), int(r["y"]), int(r["x"]) + bw, int(r["y"]) + bh)
        o, t, a = infer(model, img, box, route(bw / bh), dev)
        # rfdetr antenna boxes
        cx0, cy0, cx1, cy1 = crop_pad_box(box, H, W); carcrop = img[cy0:cy1, cx0:cx1]
        boxes = []
        if carcrop.shape[0] >= 32 and carcrop.shape[1] >= 32:
            dd = det.predict(Image.fromarray(carcrop), threshold=DET_THR)
            cls = np.asarray(dd.class_id) if getattr(dd, "class_id", None) is not None else None
            for bi, b in enumerate(np.asarray(dd.xyxy)):
                if cls is not None and int(cls[bi]) != 0: continue
                x0, y0, x1, y1 = [int(v) for v in b]
                m = np.zeros((H, W), bool); m[max(0,cy0+y0):cy0+y1, max(0,cx0+x0):cx0+x1] = True
                if m.any(): boxes.append(m)
        # head CCs
        nlab, lbl, st, _ = cv2.connectedComponentsWithStats(a.astype(np.uint8), 8)
        heads = [lbl == i for i in range(1, nlab) if st[i, cv2.CC_STAT_AREA] >= HEAD_MINPX]

        def tp(m): return (m & gt_a).sum() >= HIT
        # rfdetr proposer
        for m in boxes:
            S["rf_tp" if tp(m) else "rf_fp"] += 1
        # head proposer
        for m in heads:
            S["hd_tp" if tp(m) else "hd_fp"] += 1
        # head's ADDED candidates (don't overlap any rfdetr box) = marginal cost of union
        for m in heads:
            if not any((m & b).any() for b in boxes):
                S["add_tp" if tp(m) else "add_fp"] += 1
        # clean-image fire (no GT antenna)
        if not has_ant:
            nclean += 1
            if boxes: S["rf_fire_clean"] += 1
            if heads: S["hd_fire_clean"] += 1
            if boxes or heads: S["un_fire_clean"] += 1

    def prec(tpc, fpc): return 100 * tpc / max(1, tpc + fpc)
    print(f"\n=== ANTENNA PROPOSER FALSE-POSITIVES ({nimg} test imgs, {nclean} with NO GT antenna) ===")
    print(f"  proposer   cand   TP    FP   precision   FP/img")
    for nm, lab in (("rf", "RFDETR"), ("hd", "HEAD")):
        tpc, fpc = S[f"{nm}_tp"], S[f"{nm}_fp"]
        print(f"  {lab:<9}{tpc+fpc:>5}{tpc:>6}{fpc:>6}{prec(tpc,fpc):>9.0f}%{fpc/max(1,nimg):>9.2f}")
    print(f"\n  MARGINAL cost of adding HEAD to RFDETR (head candidates not overlapping any rfdetr box):")
    print(f"    extra TP (real antennas rfdetr missed): {S['add_tp']}   extra FP (spurious stubs): {S['add_fp']}"
          f"   -> {prec(S['add_tp'], S['add_fp']):.0f}% of added candidates are real")
    print(f"\n  CLEAN-IMAGE fire rate (fired >=1 antenna candidate on a no-antenna image):")
    print(f"    RFDETR {100*S['rf_fire_clean']/max(1,nclean):.0f}%   HEAD {100*S['hd_fire_clean']/max(1,nclean):.0f}%"
          f"   UNION {100*S['un_fire_clean']/max(1,nclean):.0f}%   (of {nclean})")


if __name__ == "__main__":
    main()
