#!/usr/bin/env python3
"""
Find & diagnose ours-v2 outline segment-drops (>2% of car) over the test set, and render
them: raw+detector box | GT | BiRefNet body | ours outline (dropped GT region in red).
Also reports whether the dropped chunk lies OUTSIDE the detector crop (-> detector clipping)
or inside (-> BiRefNet miss) — the key diagnostic.

Run from repo root:
  PYTHONPATH=.:carcutter/car_bbox_detector/birefnet/BiRefNet python carcutter/car_bbox_detector/render_ours_drops.py
"""
import argparse, csv
from pathlib import Path
import numpy as np, cv2, torch
from PIL import Image
import segmentation_models_pytorch as smp
import rfdetr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from carcutter.car_bbox_detector.gen_coarse_masks import coarse_prob
from carcutter.car_bbox_detector.antenna_continuity import colfill
from carcutter.car_bbox_detector.antenna_fp_clean import keep_main
from carcutter.car_bbox_detector.birefnet_eval import load_birefnet, birefnet_outline
from carcutter.car_bbox_detector.cascade_eval import predict_union
ANT = 1; EXP = "carcutter/car_bbox_detector/experiments"
BF = "carcutter/car_bbox_detector/birefnet/BiRefNet/ckpts/car_outline_v1/epoch_10.pth"
ANT_THR = 0.5; THR = 0.02


def load_unet(ck, dev):
    m = smp.Unet("efficientnet-b4", encoder_weights=None, in_channels=3, classes=1).to(dev).eval()
    st = torch.load(ck, map_location=dev); m.load_state_dict(st.get("model", st)); return m


def boxmask(b, H, W):
    m = np.zeros((H, W), np.uint8); x0, y0, x1, y1 = [int(v) for v in b]
    m[max(0, y0):min(H, y1 + 1), max(0, x0):min(W, x1 + 1)] = 255; return m


def biggest_miss(gt, pred):
    miss = (gt & ~pred).astype(np.uint8)
    if miss.sum() == 0: return 0.0, None
    n, lbl, st, _ = cv2.connectedComponentsWithStats(miss, 8)
    if n <= 1: return 0.0, None
    bi = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return float(st[bi, cv2.CC_STAT_AREA]) / float(gt.sum()), (lbl == bi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="carcutter/car_bbox_detector/data/index.csv")
    ap.add_argument("--out", default="carcutter/car_bbox_detector/data/ours_drops.png")
    ap.add_argument("--bf", default=BF, help="BiRefNet checkpoint")
    args = ap.parse_args()
    dev = "cuda"
    det = rfdetr.RFDETRMedium.from_checkpoint(f"{EXP}/rfdetr_car_antenna_v1/checkpoint_best_total.pth")
    bf = load_birefnet(args.bf, dev); ant = load_unet(f"{EXP}/unet_antenna_v1/checkpoints/best.pt", dev)
    rows = [r for r in csv.DictReader(open(args.index)) if r["split"] == "test"]
    bad = []
    for i, r in enumerate(rows):
        pil = Image.open(r["image"]).convert("RGB"); img = np.array(pil); H, W = img.shape[:2]
        gt = np.array(Image.open(r["mask"]).convert("RGB")).sum(2) > 0
        if gt.sum() < 20: continue
        carbox = predict_union(det, pil, 0.3, 0.15)
        if carbox is None: continue
        body = birefnet_outline(bf, img, carbox, dev)
        am = np.zeros((H, W), bool)
        d = det.predict(pil, threshold=ANT_THR)
        cls, conf, xyxy = np.asarray(d.class_id), np.asarray(d.confidence), np.asarray(d.xyxy)
        for j in range(len(xyxy)):
            if cls[j] == ANT and conf[j] >= ANT_THR:
                am |= colfill(coarse_prob(ant, img, boxmask(xyxy[j], H, W), 256, dev) > 0.5)
        outline = keep_main(body | am, 9)
        frac, miss = biggest_miss(gt, outline)
        if frac > THR:
            # is the dropped chunk outside the detector crop?
            x0, y0, x1, y1 = [int(v) for v in carbox]
            cb = np.zeros((H, W), bool); cb[max(0,y0):y1, max(0,x0):x1] = True
            out_frac = float((miss & ~cb).sum()) / float(miss.sum()) if miss is not None else 0.0
            bad.append(dict(stem=Path(r["image"]).stem, r=r, frac=frac, carbox=carbox,
                            body=body.copy(), outline=outline.copy(), gt=gt.copy(),
                            miss=miss.copy(), out_frac=out_frac))
            print(f"  DROP {frac*100:.1f}%  outside_crop={out_frac*100:.0f}%  {Path(r['image']).stem[-30:]}", flush=True)
        if (i + 1) % 100 == 0: print(f"  ..{i+1}/{len(rows)}", flush=True)

    bad.sort(key=lambda x: -x["frac"])
    print(f"\n{len(bad)} cases with outline drop > {int(THR*100)}% of car")
    if not bad: return
    n = len(bad); fig, ax = plt.subplots(n, 4, figsize=(20, 5 * n)); ax = np.atleast_2d(ax)
    for i, b in enumerate(bad):
        img = np.array(Image.open(b["r"]["image"]).convert("RGB"))
        raw = img.copy(); x0, y0, x1, y1 = [int(v) for v in b["carbox"]]
        cv2.rectangle(raw, (x0, y0), (x1, y1), (255, 255, 0), 4)
        def ov(m, col, miss=None):
            o = img.copy(); o[m] = (0.5 * o[m] + 0.5 * np.array(col)).astype(np.uint8)
            if miss is not None: o[miss] = (255, 0, 0)
            return o
        diag = "OUTSIDE crop (detector clip)" if b["out_frac"] > 0.5 else "inside crop (model miss)"
        panels = [(raw, "raw + det box"), (ov(b["gt"], [0, 220, 0]), "GT"),
                  (ov(b["body"], [0, 180, 255]), "BiRefNet body"),
                  (ov(b["outline"], [0, 180, 255], b["miss"]), f"outline drop={b['frac']*100:.1f}% [{diag}]")]
        for c, (pim, t) in enumerate(panels):
            ax[i, c].imshow(pim); ax[i, c].set_title(t, fontsize=10); ax[i, c].axis("off")
    fig.suptitle("ours-v2 outline segment-drops (>2% of car); red = dropped GT region", fontsize=14)
    fig.tight_layout(); fig.savefig(args.out, dpi=78, bbox_inches="tight"); print(f"-> {args.out}")


if __name__ == "__main__":
    main()
