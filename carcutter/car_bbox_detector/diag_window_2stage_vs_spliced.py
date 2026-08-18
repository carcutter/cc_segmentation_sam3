#!/usr/bin/env python3
"""
Decisive test of the encoder-fit hypothesis: does the tint-SPECIALIZED 2-stage window pipeline
(RF-DETR window-region detector -> tight 512 crop -> dedicated UNet) catch the dark/see-through glass
that the SPLICED shared-encoder tint head misses?

Per GT-tint test row compute tint IoU for: spliced head | 2-stage UNet | union | prod.
Report overall AND on the SPLICED-MISS subset (spliced IoU<0.3) -- if 2-stage recovers those,
a tint-specialized representation is the bottleneck (supports the hypothesis); if it also misses,
it's resolution/ambiguity.
"""
import csv, os, sys
from collections import defaultdict
import numpy as np, cv2, torch
from PIL import Image
import segmentation_models_pytorch as smp
import rfdetr
sys.path.insert(0, "carcutter/car_bbox_detector/birefnet/BiRefNet")
from carcutter.car_bbox_detector.seg_eval import mask_iou
from carcutter.car_bbox_detector.probe_birefnet_aspect import route, crop_pad_box
from carcutter.car_bbox_detector.build_spliced_eval import Spliced, load_bn, infer, PROD5, TRI

ROOT = "carcutter/car_bbox_detector"; EXP = f"{ROOT}/experiments"; BLUE = (0, 0, 255); dev = "cuda"
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
SZ = 512; DET_THR = 0.3; SEG_THR = 0.5


def binar(p):
    if not p or not os.path.exists(p): return None
    a = np.array(Image.open(p)); return (a.sum(2) > 0) if a.ndim == 3 else (a > 0)


@torch.no_grad()
def seg_box(seg, img, gbox):
    H, W = img.shape[:2]; x0, y0, x1, y1 = [max(0, int(gbox[0])), max(0, int(gbox[1])), min(W, int(gbox[2])), min(H, int(gbox[3]))]
    if x1 - x0 < 8 or y1 - y0 < 8: return None
    tile = cv2.resize(img[y0:y1, x0:x1], (SZ, SZ))
    t = torch.from_numpy(((tile / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]).float().to(dev)
    pr = (torch.sigmoid(seg(t))[0, 0] > SEG_THR).cpu().numpy().astype(np.uint8)
    pr = cv2.resize(pr, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST) > 0
    full = np.zeros((H, W), bool); full[y0:y1, x0:x1] = pr; return full


def main():
    model = Spliced(load_bn(PROD5, 1, dev), load_bn(TRI, 3, dev)).eval()
    det = rfdetr.RFDETRMedium.from_checkpoint(f"{EXP}/rfdetr_windowcrop_v1/checkpoint_best_ema.pth")
    seg = smp.Unet("efficientnet-b2", encoder_weights=None, in_channels=3, classes=1).to(dev).eval()
    seg.load_state_dict(torch.load(f"{EXP}/unet_windowseg_v2/best.pt", map_location=dev)["model"])
    rows = [r for r in csv.DictReader(open(f"{ROOT}/data/index.csv")) if r["split"] == "test"]

    agg = {b: defaultdict(list) for b in ("all", "miss", "prodwin")}
    n = 0
    for r in rows:
        rgb = np.array(Image.open(r["mask"]).convert("RGB")); gt_t = (rgb == BLUE).all(2)
        if gt_t.sum() < 60: continue
        n += 1
        img = np.array(Image.open(r["image"]).convert("RGB")); H, W = img.shape[:2]
        bw, bh = int(r["bw"]), int(r["bh"]); box = (int(r["x"]), int(r["y"]), int(r["x"]) + bw, int(r["y"]) + bh)
        o, t, a = infer(model, img, box, route(bw / bh), dev)
        # 2-stage: window-region detector on padded car crop -> UNet per box
        cx0, cy0, cx1, cy1 = crop_pad_box(box, H, W); carcrop = img[cy0:cy1, cx0:cx1]
        ts = np.zeros((H, W), bool)
        if carcrop.shape[0] >= 32 and carcrop.shape[1] >= 32:
            dd = det.predict(Image.fromarray(carcrop), threshold=DET_THR)
            for b in np.asarray(dd.xyxy):
                m = seg_box(seg, img, (cx0 + b[0], cy0 + b[1], cx0 + b[2], cy0 + b[3]))
                if m is not None: ts |= m
        uni = t | ts
        prod = binar(r["prod_outline"].replace("/outline/", "/holes_tint/")) if r["prod_outline"] else None
        i_sp, i_ts, i_un = mask_iou(gt_t, t), mask_iou(gt_t, ts), mask_iou(gt_t, uni)
        i_pr = mask_iou(gt_t, prod) if prod is not None else float("nan")
        buckets = ["all"]
        if i_sp < 0.3:
            buckets.append("miss")
            if not np.isnan(i_pr) and i_pr >= 0.6: buckets.append("prodwin")  # prod clearly wins, we miss
        for b in buckets:
            agg[b]["sp"].append(i_sp); agg[b]["ts"].append(i_ts); agg[b]["un"].append(i_un); agg[b]["pr"].append(i_pr)
        if n % 150 == 0: print(f"  {n}", flush=True)

    print(f"\nGT-tint test rows: {n}   (tint IoU; 2stage = RF-DETR windowcrop + UNet)")
    print(f"  {'subset':<14}{'n':>5}{'SPLICED':>9}{'2STAGE':>8}{'UNION':>8}{'PROD':>8}")
    for b, lab in (("all", "ALL"), ("miss", "spliced<0.3"), ("prodwin", "sp<0.3 & prod>=0.6")):
        d = agg[b]
        if d["sp"]:
            print(f"  {lab:<14}{len(d['sp']):>5}{np.mean(d['sp']):>9.2f}{np.mean(d['ts']):>8.2f}"
                  f"{np.mean(d['un']):>8.2f}{np.nanmean(d['pr']):>8.2f}")
    print("\n  If 2STAGE >> SPLICED on the spliced<0.3 subset => tint-specialized representation recovers")
    print("  what the outline-trained frozen encoder misses (supports encoder-fit hypothesis); UNION is the fix.")


if __name__ == "__main__":
    main()
