#!/usr/bin/env python3
"""
Re-score the end-to-end comparison against the CORRECTED production references.
Reuses cached OURS masks (data/e2e/masks/<stem>.npz) — no GPU needed.

Corrected prod references (per domain owner):
  - outline : prod 'outline' is a filled convex-hull that fills in the genuine
              see-through gaps; subtract 'holes_punchout' to recover the same
              semantic car surface as GT (rgb.sum>0) and ours (body∪antenna).
              GT and ours need NO change.
  - holes   : our GT blue = TINTED windows = prod 'holes_tint' (NOT 'holes',
              which also bundles the see-through punchout). Compare to holes_tint.
  - antenna : prod has no antenna class; report how much each method's OUTLINE
              covers the GT antenna (recall) — ours captures antennas by
              construction, prod clips them. Also report our dedicated antenna
              mask quality (IoU / BF@1 vs GT white).

Writes data/e2e/e2e_metrics_corrected.csv and prints aggregates.

Run from repo root:
  /home/rutger/miniconda3/envs/cc_sam3/bin/python carcutter/car_bbox_detector/e2e_rescore.py
"""
import argparse, csv, os
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
from carcutter.car_bbox_detector.seg_eval import boundary_f, mask_iou

BLUE, WHITE = (0, 0, 255), (255, 255, 255)


def binar(p):
    if not p or not os.path.exists(p):
        return None
    a = np.array(Image.open(p))
    return (a.sum(2) > 0) if a.ndim == 3 else (a > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2e", default="carcutter/car_bbox_detector/data/e2e")
    ap.add_argument("--index", default="carcutter/car_bbox_detector/data/index.csv")
    args = ap.parse_args()
    e2e = Path(args.e2e)
    idx = {Path(r["image"]).stem: r for r in csv.DictReader(open(args.index))}

    fout = open(e2e / "e2e_metrics_corrected.csv", "w", newline="")
    wr = csv.writer(fout); wr.writerow(["image", "cls", "method", "iou", "bf1", "bf2", "bf3"])
    agg = defaultdict(lambda: defaultdict(list))
    ant_cover = {"ours": [], "prod": []}

    def log(img, cls, method, gt, pr):
        row = [mask_iou(gt, pr), boundary_f(gt, pr, 1), boundary_f(gt, pr, 2), boundary_f(gt, pr, 3)]
        wr.writerow([img, cls, method] + [f"{x:.4f}" for x in row])
        for k, v in zip(("iou", "bf1", "bf2", "bf3"), row):
            agg[(cls, method)][k].append(v)

    for npz in sorted((e2e / "masks").glob("*.npz")):
        stem = npz.stem
        r = idx.get(stem)
        if r is None:
            continue
        d = np.load(npz)
        rgb = np.array(Image.open(r["mask"]).convert("RGB"))
        gt = {"outline": rgb.sum(2) > 0, "holes": (rgb == BLUE).all(2), "antenna": (rgb == WHITE).all(2)}
        ours = {"outline": d["outline"].astype(bool), "holes": d["holes"].astype(bool),
                "antenna": d["antenna"].astype(bool)}
        po = r["prod_outline"]
        has_prod = bool(po) and os.path.exists(po)

        # OUTLINE — ours vs GT; prod = outline - punchout
        if gt["outline"].sum() >= 20:
            log(r["image"], "outline", "ours", gt["outline"], ours["outline"])
            if has_prod:
                ol = binar(po); hp = binar(po.replace("/outline/", "/holes_punchout/"))
                prod_ol = (ol & ~hp) if (ol is not None and hp is not None) else ol
                if prod_ol is not None:
                    log(r["image"], "outline", "prod", gt["outline"], prod_ol)

        # HOLES — ours vs GT blue; prod = holes_tint
        if gt["holes"].sum() >= 20:
            log(r["image"], "holes", "ours", gt["holes"], ours["holes"])
            if has_prod:
                ht = binar(po.replace("/outline/", "/holes_tint/"))
                if ht is not None:
                    log(r["image"], "holes", "prod", gt["holes"], ht)

        # ANTENNA — ours mask quality vs GT white; outline-coverage (recall) ours vs prod
        if gt["antenna"].sum() >= 20:
            log(r["image"], "antenna", "ours", gt["antenna"], ours["antenna"])
            w = gt["antenna"]; denom = w.sum()
            ant_cover["ours"].append((w & ours["outline"]).sum() / denom)
            if has_prod:
                ol = binar(po)
                if ol is not None:
                    ant_cover["prod"].append((w & ol).sum() / denom)

    fout.close()
    m = lambda a: float(np.mean(a)) if a else float("nan")
    print("CORRECTED end-to-end comparison (ours vs prod):\n")
    print(f"  {'class':<9}{'method':<6}{'n':>5}{'IoU':>9}{'BF@1':>9}{'BF@2':>9}{'BF@3':>9}")
    for cls in ("outline", "holes", "antenna"):
        for meth in ("ours", "prod"):
            a = agg[(cls, meth)]
            if not a:
                continue
            print(f"  {cls:<9}{meth:<6}{len(a['iou']):>5}" +
                  "".join(f"{m(a[k]):>9.3f}" for k in ("iou", "bf1", "bf2", "bf3")))
        print()
    print("ANTENNA outline-coverage (recall of GT antenna inside the silhouette):")
    print(f"  ours outline covers {100*m(ant_cover['ours']):.1f}% of antenna pixels (n={len(ant_cover['ours'])})")
    print(f"  prod outline covers {100*m(ant_cover['prod']):.1f}% of antenna pixels (n={len(ant_cover['prod'])})")
    print(f"\n-> {e2e/'e2e_metrics_corrected.csv'}")


if __name__ == "__main__":
    main()
