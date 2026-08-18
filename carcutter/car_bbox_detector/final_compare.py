#!/usr/bin/env python3
"""
Final ours-v2 (BiRefNet hybrid) vs ours-v1 (UNet) vs prod comparison, all classes,
on the e2e test set. CPU-only: reuses cached UNet masks (data/e2e/masks/<stem>.npz),
cached BiRefNet body masks (data/e2e/birefnet_outline/<stem>.npy), and optionally
cached BiRefNet holes (data/e2e/birefnet_holes/<stem>.npy).

OUTLINE variants (vs GT union): prod (outline-punchout), unet (body∪antenna), birefnet
(body only), hybrid (birefnet_body ∪ antenna_unet, antenna FP-cleaned via keep_main).
Also antenna coverage (recall of GT antenna inside each outline).
HOLES variants (vs GT blue): prod (holes_tint), unet, birefnet (if cached).

Run from repo root:
  PYTHONPATH=. python carcutter/car_bbox_detector/final_compare.py
"""
import argparse, csv, os
from pathlib import Path
from collections import defaultdict
import numpy as np, cv2
from PIL import Image
from carcutter.car_bbox_detector.seg_eval import boundary_f, mask_iou
from carcutter.car_bbox_detector.antenna_fp_clean import keep_main
BLUE, WHITE = (0, 0, 255), (255, 255, 255)


def binar(p):
    if not p or not os.path.exists(p):
        return None
    a = np.array(Image.open(p)); return (a.sum(2) > 0) if a.ndim == 3 else (a > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2e", default="carcutter/car_bbox_detector/data/e2e")
    ap.add_argument("--index", default="carcutter/car_bbox_detector/data/index.csv")
    ap.add_argument("--clean-close", type=int, default=9, help="antenna FP keep_main close px")
    args = ap.parse_args()
    idx = {Path(r["image"]).stem: r for r in csv.DictReader(open(args.index))}
    bf_out = Path(args.e2e) / "birefnet_outline"
    bf_hol = Path(args.e2e) / "birefnet_holes"

    o = defaultdict(lambda: defaultdict(list))   # outline: method -> metric -> list
    ac = defaultdict(list)                        # antenna coverage
    h = defaultdict(lambda: defaultdict(list))    # holes
    for npz in sorted((Path(args.e2e) / "masks").glob("*.npz")):
        r = idx.get(npz.stem)
        if r is None:
            continue
        d = np.load(npz)
        rgb = np.array(Image.open(r["mask"]).convert("RGB"))
        gtu = rgb.sum(2) > 0; gtb = (rgb == BLUE).all(2); gtw = (rgb == WHITE).all(2)
        unet_out = d["outline"].astype(bool); antenna = d["antenna"].astype(bool)
        po = r["prod_outline"]; has_prod = bool(po) and os.path.exists(po)

        # ---- OUTLINE ----
        outs = {"unet": unet_out}
        bfp = bf_out / f"{npz.stem}.npy"
        if bfp.exists():
            body = np.load(bfp)
            outs["birefnet"] = body
            hyb = body | antenna
            outs["hybrid"] = keep_main(hyb, args.clean_close)   # FP-clean the union
        if has_prod:
            ol = binar(po); hp = binar(po.replace("/outline/", "/holes_punchout/"))
            outs["prod"] = (ol & ~hp) if (ol is not None and hp is not None) else ol
        if gtu.sum() >= 20:
            for k, m in outs.items():
                if m is None:
                    continue
                o[k]["iou"].append(mask_iou(gtu, m))
                for t in (1, 2, 3):
                    o[k][f"bf{t}"].append(boundary_f(gtu, m, t))
                if gtw.sum() >= 20:
                    ac[k].append((gtw & m).sum() / gtw.sum())

        # ---- HOLES ----
        if gtb.sum() >= 20:
            hs = {"unet": d["holes"].astype(bool)}
            bhp = bf_hol / f"{npz.stem}.npy"
            if bhp.exists():
                hs["birefnet"] = np.load(bhp)
            if has_prod:
                hs["prod"] = binar(po.replace("/outline/", "/holes_tint/"))
            for k, m in hs.items():
                if m is None:
                    continue
                h[k]["iou"].append(mask_iou(gtb, m))
                for t in (1, 2, 3):
                    h[k][f"bf{t}"].append(boundary_f(gtb, m, t))

    m = lambda a: float(np.mean(a)) if a else float("nan")
    def tbl(title, agg, methods, extra=None):
        print(f"\n{title}")
        print(f"  {'method':<10}{'n':>5}{'IoU':>9}{'BF@1':>9}{'BF@2':>9}{'BF@3':>9}" + (f"{'ant_cov':>9}" if extra else ""))
        for k in methods:
            a = agg[k]
            if not a["iou"]:
                continue
            row = f"  {k:<10}{len(a['iou']):>5}" + "".join(f"{m(a[x]):>9.3f}" for x in ("iou", "bf1", "bf2", "bf3"))
            if extra is not None:
                row += f"{m(extra[k]):>9.3f}" if extra[k] else f"{'-':>9}"
            print(row)
    tbl("OUTLINE vs GT union:", o, ["prod", "unet", "birefnet", "hybrid"], extra=ac)
    tbl("HOLES vs GT blue (prod=holes_tint):", h, ["prod", "unet", "birefnet"])


if __name__ == "__main__":
    main()
