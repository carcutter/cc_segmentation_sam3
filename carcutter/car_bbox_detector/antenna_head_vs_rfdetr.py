#!/usr/bin/env python3
"""
Is the BiRefNet antenna HEAD (car-crop level) a complementary/better antenna PROPOSER than the
RF-DETR antenna detector? Per GT antenna instance, does each localize it?
  rfdetr_hit : GT-antenna px inside any rfdetr antenna box (class0) >= thr
  head_hit   : GT-antenna px covered by spliced head mask `a` >= thr
Report instance recall for each + complementary sets (head-only = head finds, rfdetr misses; and vv),
overall + thin-mast subset. If head-only is sizeable => head proposes antennas rfdetr can't, worth
feeding the zoom-in UNet from head candidates (not a dumb line-join).
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
DET_THR = 0.25  # lenient -> maximize rfdetr recall so head-only set is conservative
MINPX = 20


def thin(comp):
    ys, xs = np.where(comp)
    h = ys.max() - ys.min() + 1; w = xs.max() - xs.min() + 1; L = max(h, w)
    return (L >= 22) and (len(ys) / L <= 6.0) and (len(ys) / float(h * w) <= 0.35)


def main():
    model = Spliced(load_bn(PROD5, 1, dev), load_bn(TRI, 3, dev)).eval()
    det = rfdetr.RFDETRMedium.from_checkpoint(f"{EXP}/rfdetr_multiclass_v1/checkpoint_best_ema.pth")
    rows = [r for r in csv.DictReader(open(f"{ROOT}/data/index.csv")) if r["split"] == "test"]

    THR = [0.1, 0.3]
    C = {t: {sub: defaultdict(int) for sub in ("all", "thin")} for t in THR}  # counts: gt, rf, hd, both, hdonly, rfonly
    for r in rows:
        rgb = np.array(Image.open(r["mask"]).convert("RGB")); gt_a = (rgb == WHITE).all(2)
        if gt_a.sum() < MINPX: continue
        img = np.array(Image.open(r["image"]).convert("RGB")); H, W = img.shape[:2]
        bw, bh = int(r["bw"]), int(r["bh"]); box = (int(r["x"]), int(r["y"]), int(r["x"]) + bw, int(r["y"]) + bh)
        o, t, a = infer(model, img, box, route(bw / bh), dev)
        # rfdetr antenna boxes (class 0) on padded car crop -> full-image box mask
        cx0, cy0, cx1, cy1 = crop_pad_box(box, H, W); carcrop = img[cy0:cy1, cx0:cx1]
        rfmask = np.zeros((H, W), bool)
        if carcrop.shape[0] >= 32 and carcrop.shape[1] >= 32:
            dd = det.predict(Image.fromarray(carcrop), threshold=DET_THR)
            cls = np.asarray(dd.class_id) if getattr(dd, "class_id", None) is not None else None
            for bi, b in enumerate(np.asarray(dd.xyxy)):
                if cls is not None and int(cls[bi]) != 0: continue  # 0 = antenna
                x0, y0, x1, y1 = [int(v) for v in b]
                rfmask[max(0,cy0+y0):cy0+y1, max(0,cx0+x0):cx0+x1] = True
        # per GT antenna instance
        nlab, lbl, st, _ = cv2.connectedComponentsWithStats(gt_a.astype(np.uint8), 8)
        for i in range(1, nlab):
            comp = lbl == i; area = st[i, cv2.CC_STAT_AREA]
            if area < MINPX: continue
            isthin = thin(comp)
            rf_f = (comp & rfmask).sum() / area
            hd_f = (comp & a).sum() / area
            for tt in THR:
                rf = rf_f >= tt; hd = hd_f >= tt
                for sub in (("all",) + (("thin",) if isthin else ())):
                    c = C[tt][sub]; c["gt"] += 1; c["rf"] += rf; c["hd"] += hd
                    c["both"] += rf and hd; c["hdonly"] += hd and not rf; c["rfonly"] += rf and not hd

    for tt in THR:
        print(f"\n=== ANTENNA PROPOSER recall @ overlap>={tt} (rfdetr antenna det_thr={DET_THR}) ===")
        print(f"  {'subset':<6}{'GT#':>6}{'RFDETR':>9}{'HEAD':>8}{'both':>7}{'HEAD-only':>11}{'RFDETR-only':>13}")
        for sub in ("all", "thin"):
            c = C[tt][sub]; g = max(1, c["gt"])
            print(f"  {sub:<6}{c['gt']:>6}{100*c['rf']/g:>8.0f}%{100*c['hd']/g:>7.0f}%{100*c['both']/g:>6.0f}%"
                  f"{c['hdonly']:>7}({100*c['hdonly']/g:.0f}%){c['rfonly']:>6}({100*c['rfonly']/g:.0f}%)")
    print("\n  HEAD-only > 0 => head proposes antennas rfdetr misses (worth feeding zoom-UNet from head candidates).")


if __name__ == "__main__":
    main()
