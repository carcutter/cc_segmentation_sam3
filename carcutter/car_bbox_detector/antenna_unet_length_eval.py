#!/usr/bin/env python3
"""
Does the ZOOM-IN antenna UNet (EffNet-B4 @256, deploy path) actually get thin 1px masts, or does its
internal /32 downsampling kill them too? Same metric as the BiRefNet head eval: thin-mast TIP/MID/BASE
length coverage + continuity (fragmented / tip-loss / clean). Favorable framing: crop = 1.5x the GT
antenna bbox (coarse_prob), isolating the UNet's resolution capability (not the proposer).
Measure UNet RAW vs +COLFILL (deployed; colfill = per-column vertical gap-fill = baked-in completion).
Ref (spliced BiRefNet): head tip71/mid66/base46 ; o|head tip79/mid82/base88.
"""
import csv, sys
from collections import defaultdict
import numpy as np, cv2, torch
from PIL import Image
from scipy.ndimage import binary_dilation
import segmentation_models_pytorch as smp
from carcutter.car_bbox_detector.gen_coarse_masks import coarse_prob
from carcutter.car_bbox_detector.antenna_continuity import colfill

ROOT = "carcutter/car_bbox_detector"; EXP = f"{ROOT}/experiments"; WHITE = (255, 255, 255); dev = "cuda"


def thin(gt_a):
    ys, xs = np.where(gt_a)
    if len(ys) < 8: return False, ys, xs
    h = ys.max() - ys.min() + 1; w = xs.max() - xs.min() + 1; L = max(h, w)
    return (L >= 22) and (len(ys) / L <= 6.0) and (len(ys) / float(h * w) <= 0.35), ys, xs


def thirds_cov(gt_a, pred, ys, tol):
    pd = binary_dilation(pred, iterations=tol)
    y0, y1 = ys.min(), ys.max(); edges = np.linspace(y0, y1 + 1, 4); rowidx = np.arange(gt_a.shape[0])[:, None]
    out = []
    for k in range(3):
        seg = gt_a & (rowidx >= edges[k]) & (rowidx < edges[k + 1]); s = seg.sum()
        out.append((seg & pd).sum() / s if s else float("nan"))
    return out


def continuity(gt_a, pred, ys, tol):
    pd = binary_dilation(pred, iterations=tol)
    seq = [bool((gt_a[y] & pd[y]).any()) for y in range(ys.min(), ys.max() + 1) if gt_a[y].any()]
    seq = np.array(seq)
    if seq.size == 0: return "miss"
    lead = 0
    for v in seq:
        if v: break
        lead += 1
    ones = np.where(seq)[0]
    internal = any((b - a > 1) for a, b in zip(ones[:-1], ones[1:])) if len(ones) >= 2 else False
    if internal: return "frag"
    if lead > 0 or not seq.all(): return "tiploss"
    return "clean"


def main():
    ant = smp.Unet("efficientnet-b4", encoder_weights=None, in_channels=3, classes=1).to(dev).eval()
    st = torch.load(f"{EXP}/unet_antenna_v1/checkpoints/best.pt", map_location=dev); ant.load_state_dict(st.get("model", st))
    rows = [r for r in csv.DictReader(open(f"{ROOT}/data/index.csv")) if r["split"] == "test"]
    agg = {k: {"tip": [], "mid": [], "base": [], "cls": defaultdict(int)} for k in ("raw", "colfill")}
    n = 0
    for r in rows:
        gt_a = (np.array(Image.open(r["mask"]).convert("RGB")) == WHITE).all(2)
        if gt_a.sum() < 20: continue
        ist, ys, xs = thin(gt_a)
        if not ist: continue
        n += 1
        img = np.array(Image.open(r["image"]).convert("RGB"))
        prob = coarse_prob(ant, img, (gt_a * 255).astype(np.uint8), 256, dev)
        raw = prob > 0.5
        L = max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1); tol = max(2, int(round(0.03 * L)))
        for nm, pred in (("raw", raw), ("colfill", colfill(raw))):
            tp, md, bs = thirds_cov(gt_a, pred, ys, tol)
            agg[nm]["tip"].append(tp); agg[nm]["mid"].append(md); agg[nm]["base"].append(bs)
            agg[nm]["cls"][continuity(gt_a, pred, ys, max(1, int(round(0.02 * L))))] += 1
        if n % 40 == 0: print(f"  {n}", flush=True)

    print(f"\nZOOM-IN ANTENNA UNET on thin masts (n={n}, favorable GT-bbox crop, EffNet-B4 @256):")
    print(f"  {'pred':<10}{'TIP':>6}{'MID':>6}{'BASE':>6}   {'clean':>7}{'frag':>6}{'tiploss':>9}{'miss':>6}")
    for nm in ("raw", "colfill"):
        d = agg[nm]; c = d["cls"]; g = max(1, n)
        print(f"  unet-{nm:<5}{np.nanmean(d['tip'])*100:>5.0f}%{np.nanmean(d['mid'])*100:>5.0f}%{np.nanmean(d['base'])*100:>5.0f}%"
              f"   {100*c['clean']/g:>6.0f}%{100*c['frag']/g:>5.0f}%{100*c['tiploss']/g:>8.0f}%{100*c['miss']/g:>5.0f}%")
    print(f"  REF spliced BiRefNet: head tip71/mid66/base46 ; o|head tip79/mid82/base88 (clean47/frag23/tiploss30).")
    print("  If unet-raw TIP is low too => the UNet's /32 downsampling ALSO kills the thin tip, zoom doesn't save it.")


if __name__ == "__main__":
    main()
