#!/usr/bin/env python3
"""
A/B: does the 1536 body BiRefNet pass add structural-hole recall beyond 1024 + 2-stage?
The 1536 pass exists ONLY to let small punchout holes register in the silhouette; the Phase A
2-stage hole pipeline now also recovers that small tail. So compare instance-level structural-hole
recall by size for:
  body1024-only | body1024 ∪ 2stage | body1536-only | body1536 ∪ 2stage
If 1024∪2stage matches 1536∪2stage on small holes, the second BiRefNet pass can be dropped.

Run from repo root:
  PYTHONPATH=. /home/rutger/miniconda3/envs/cc_sam3/bin/python carcutter/car_bbox_detector/eval_body_res_ab.py --n 200
"""
import argparse, csv
from pathlib import Path
import numpy as np, cv2, torch
from PIL import Image
from scipy.ndimage import binary_fill_holes
import segmentation_models_pytorch as smp
import rfdetr
from carcutter.car_bbox_detector.seg_eval import MEAN, STD
from carcutter.car_bbox_detector.gen_coarse_masks import coarse_prob
from carcutter.car_bbox_detector.antenna_continuity import colfill
from carcutter.car_bbox_detector.cascade_eval import predict_union
from carcutter.car_bbox_detector.birefnet_eval import load_birefnet, birefnet_outline
from carcutter.car_bbox_detector.antenna_fp_clean import keep_main

ROOT = "carcutter/car_bbox_detector"; EXP = f"{ROOT}/experiments"
BIREFNET_CKPT = f"{ROOT}/birefnet/BiRefNet/ckpts/car_outline_v2/epoch_16.pth"
MINPX = 25; ANTENNA, HOLE = 0, 1
DET_THR = 0.5; HOLE_THR = 0.6; SEG_REG = 512
SIZES = [("small", 0, 0.005), ("med", 0.005, 0.02), ("large", 0.02, 1.0)]


def load_unet(ckpt, enc, dev):
    m = smp.Unet(enc, encoder_weights=None, in_channels=3, classes=1).to(dev).eval()
    st = torch.load(ckpt, map_location=dev); m.load_state_dict(st.get("model", st)); return m


def boxmask(box, H, W):
    m = np.zeros((H, W), np.uint8); x0, y0, x1, y1 = [int(v) for v in box]
    m[max(0,y0):min(H,y1+1), max(0,x0):min(W,x1+1)] = 255; return m


@torch.no_grad()
def segment_box(seg, img, box, dev, thr, sz):
    H, W = img.shape[:2]; x0, y0, x1, y1 = [int(v) for v in box]
    x0, y0 = max(0, x0), max(0, y0); x1, y1 = min(W, x1), min(H, y1)
    if x1 - x0 < 8 or y1 - y0 < 8: return None, None
    tile = cv2.resize(img[y0:y1, x0:x1], (sz, sz))
    t = torch.from_numpy(((tile / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]).float().to(dev)
    pr = (torch.sigmoid(seg(t))[0, 0] > thr).cpu().numpy().astype(np.uint8)
    pr = cv2.resize(pr, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST) > 0
    return pr, (x0, y0, x1, y1)


def comps(mask, ca, lo, hi):
    n, lbl, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return [(lbl == i, st[i, cv2.CC_STAT_AREA]) for i in range(1, n)
            if MINPX <= st[i, cv2.CC_STAT_AREA] and lo <= st[i, cv2.CC_STAT_AREA] / ca < hi]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=f"{ROOT}/data/index.csv")
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args(); dev = "cuda"
    cardet = rfdetr.RFDETRMedium.from_checkpoint(f"{EXP}/rfdetr_car_antenna_v1/checkpoint_best_total.pth")
    unidet = rfdetr.RFDETRMedium.from_checkpoint(f"{EXP}/rfdetr_multiclass_v1/checkpoint_best_ema.pth")
    bfbody = load_birefnet(BIREFNET_CKPT, dev)
    ant = load_unet(f"{EXP}/unet_antenna_v1/checkpoints/best.pt", "efficientnet-b4", dev)
    holeseg = load_unet(f"{EXP}/unet_holeseg_v2/best.pt", "efficientnet-b2", dev)
    rows = [r for r in csv.DictReader(open(args.index)) if r["split"] == "test"][:args.n]

    KEYS = ["1024", "1024+2s", "1536", "1536+2s"]
    cnt = {s[0]: {k: 0 for k in ["GT"] + KEYS} for s in SIZES}
    prec = {k: [0, 0] for k in KEYS}                  # [real, produced]
    for i, r in enumerate(rows):
        pil = Image.open(r["image"]).convert("RGB"); img = np.array(pil); H, W = img.shape[:2]
        rgb = np.array(Image.open(r["mask"]).convert("RGB")); union = rgb.sum(2) > 0
        ca = float(union.sum())
        if ca < 1500: continue
        allholes = binary_fill_holes(union) & ~union
        carbox = predict_union(cardet, pil, 0.3, 0.15)
        if carbox is None: continue
        cx0, cy0, cx1, cy1 = [int(v) for v in carbox]
        cx0, cy0 = max(0, cx0), max(0, cy0); cx1, cy1 = min(W, cx1), min(H, cy1)
        carcrop = img[cy0:cy1, cx0:cx1]
        if carcrop.shape[0] < 32 or carcrop.shape[1] < 32: continue
        # unified detector pass -> antenna + 2-stage holes
        dd = unidet.predict(Image.fromarray(carcrop), threshold=DET_THR)
        cls = np.asarray(dd.class_id) if getattr(dd, "class_id", None) is not None else np.zeros(len(dd.xyxy))
        ant_mask = np.zeros((H, W), bool); hole2 = np.zeros((H, W), bool)
        for bi, b in enumerate(np.asarray(dd.xyxy)):
            bx0, by0, bx1, by1 = [int(v) for v in b]
            gbox = (cx0 + bx0, cy0 + by0, cx0 + bx1, cy0 + by1)
            c = int(cls[bi])
            if c == ANTENNA:
                ant_mask |= colfill(coarse_prob(ant, img, boxmask(gbox, H, W), 256, dev) > 0.5)
            elif c == HOLE:
                pr, bb = segment_box(holeseg, img, gbox, dev, HOLE_THR, SEG_REG)
                if pr is not None: x0, y0, x1, y1 = bb; hole2[y0:y1, x0:x1] |= pr
        # two body resolutions -> structural punchout each
        po = {}
        for res, tag in [(1024, "1024"), (1536, "1536")]:
            sil = keep_main(birefnet_outline(bfbody, img, carbox, dev, size=res) | ant_mask, 9)
            body_po = binary_fill_holes(sil) & ~sil
            po[tag] = body_po
            po[tag + "+2s"] = body_po | hole2
        # count instance recall by size + precision
        for name, lo, hi in SIZES:
            gtc = comps(allholes, ca, lo, hi)
            cnt[name]["GT"] += len(gtc)
            for k in KEYS:
                for comp, area in gtc:
                    if (comp & po[k]).sum() >= 0.3 * area: cnt[name][k] += 1
        for k in KEYS:
            for comp, area in comps(po[k], ca, 0, 1.0):
                prec[k][1] += 1; prec[k][0] += int((comp & allholes).sum() >= 0.3 * area)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)

    print("\nStructural punchout-hole recall by size (body-res A/B):")
    print(f"  {'size':<7}{'GT#':>6}{'1024':>9}{'1024+2s':>9}{'1536':>9}{'1536+2s':>9}")
    tot = {k: 0 for k in ["GT"] + KEYS}
    for name, _, _ in SIZES:
        row = cnt[name]; g = row["GT"]
        for k in ["GT"] + KEYS: tot[k] += row[k]
        f = lambda k: f"{100*row[k]/g:.0f}%" if g else "-"
        print(f"  {name:<7}{g:>6}{f('1024'):>9}{f('1024+2s'):>9}{f('1536'):>9}{f('1536+2s'):>9}")
    g = max(1, tot["GT"])
    print(f"  {'TOTAL':<7}{tot['GT']:>6}" + "".join(f"{100*tot[k]/g:>8.0f}%" for k in KEYS))
    print("  precision: " + "  ".join(f"{k} {100*prec[k][0]/max(1,prec[k][1]):.0f}%" for k in KEYS))


if __name__ == "__main__":
    main()
