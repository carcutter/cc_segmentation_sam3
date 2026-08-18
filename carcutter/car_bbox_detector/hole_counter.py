#!/usr/bin/env python3
"""
HOLE COUNTER (instance-level), not pixel IoU. Two hole types:
  - TINT windows      = GT blue connected components (glass; what we model)
  - STRUCTURAL holes  = topological holes of the GT car silhouette (background visible
                        THROUGH the car: roof-rail/wheel-spoke/under-spoiler gaps).
                        We never modeled these; prod's holes_punchout does.
Reports, by size bucket, how many GT holes exist and how many OURS vs PROD detect
(instance hit = >=30% of the GT hole covered).

Also verifies the prod-architecture hypothesis: holes_tint ?= holes ∩ windows,
holes_punchout ?= holes - windows.

Run from repo root:
  PYTHONPATH=. python carcutter/car_bbox_detector/hole_counter.py
"""
import argparse, csv, os
from pathlib import Path
import numpy as np, cv2
from PIL import Image
from scipy.ndimage import binary_fill_holes
BLUE = (0, 0, 255)
MINPX = 40            # ignore specks
SIZES = [("small", 0, 0.005), ("med", 0.005, 0.02), ("large", 0.02, 1.0)]   # frac of car area


def binar(p):
    if not p or not os.path.exists(p): return None
    a = np.array(Image.open(p)); return (a.sum(2) > 0) if a.ndim == 3 else (a > 0)


def comps(mask):
    n, lbl, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return [(lbl == i, int(st[i, cv2.CC_STAT_AREA])) for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= MINPX]


def iou(a, b):
    u = (a | b).sum(); return (a & b).sum() / u if u else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2e", default="carcutter/car_bbox_detector/data/e2e_v2")
    ap.add_argument("--index", default="carcutter/car_bbox_detector/data/index.csv")
    args = ap.parse_args()
    idx = {Path(r["image"]).stem: r for r in csv.DictReader(open(args.index))}

    # counts[type][size] = [n_gt, n_ours_hit, n_prod_hit]
    counts = {t: {s[0]: [0, 0, 0] for s in SIZES} for t in ("tint", "struct")}
    arch = {"tint_vs_h&w": [], "punch_vs_h-w": [], "tint_in_win": []}
    ours_prec = [0, 0]   # [our struct-holes that hit a GT hole, total our struct-holes]
    nimg = 0
    for npz in sorted((Path(args.e2e) / "masks").glob("*.npz")):
        r = idx.get(npz.stem)
        if r is None: continue
        rgb = np.array(Image.open(r["mask"]).convert("RGB"))
        union = rgb.sum(2) > 0; blue = (rgb == BLUE).all(2)
        car_area = float(union.sum())
        if car_area < 1000: continue
        nimg += 1
        d = np.load(npz)
        our_outline = d["outline"].astype(bool)
        if "punchout" in d:                                            # dedicated hi-res punchout channel
            ours_struct = d["punchout"].astype(bool)
        else:
            ours_struct = binary_fill_holes(our_outline) & ~our_outline
        po = r["prod_outline"]; hasp = bool(po) and os.path.exists(po)
        prod_tint = binar(po.replace("/outline/", "/holes_tint/")) if hasp else None
        prod_h = binar(po.replace("/outline/", "/holes/")) if hasp else None
        prod_punch = binar(po.replace("/outline/", "/holes_punchout/")) if hasp else None
        prod_win = binar(po.replace("/outline/", "/windows/")) if hasp else None
        # prod architecture check
        if hasp and all(m is not None for m in (prod_tint, prod_h, prod_punch, prod_win)):
            arch["tint_vs_h&w"].append(iou(prod_tint, prod_h & prod_win))
            arch["punch_vs_h-w"].append(iou(prod_punch, prod_h & ~prod_win))
            if prod_tint.sum() > 0:
                arch["tint_in_win"].append(float((prod_tint & prod_win).sum()) / float(prod_tint.sum()))
        # GT holes: tint windows (blue CCs) + structural (topological holes of silhouette)
        struct = binary_fill_holes(union) & ~union
        def bucket(area):
            f = area / car_area
            for name, lo, hi in SIZES:
                if lo <= f < hi: return name
            return "large"
        for gtmask_src, typ, ours_pred, prod_pred in (
                (blue, "tint", d["holes"].astype(bool), prod_tint),
                (struct, "struct", ours_struct, prod_punch)):  # ours = holes in OUR BiRefNet silhouette
            for comp, area in comps(gtmask_src):
                b = bucket(area); counts[typ][b][0] += 1
                if ours_pred is not None and (comp & ours_pred).sum() >= 0.30 * area:
                    counts[typ][b][1] += 1
                if prod_pred is not None and (comp & prod_pred).sum() >= 0.30 * area:
                    counts[typ][b][2] += 1
        # precision of OUR silhouette holes: how many are real (overlap a GT structural hole)
        for comp, area in comps(ours_struct):
            ours_prec[1] += 1
            if (comp & struct).sum() >= 0.30 * area:
                ours_prec[0] += 1

    m = lambda a: float(np.mean(a)) if a else float("nan")
    print(f"Hole counter over {nimg} test images.\n")
    print(f"PROD architecture check (user hypothesis: tint=holes∩windows, punchout=holes−windows):")
    print(f"  IoU(holes_tint, holes∩windows)   = {m(arch['tint_vs_h&w']):.3f}")
    print(f"  IoU(holes_punchout, holes−windows)= {m(arch['punch_vs_h-w']):.3f}")
    print(f"  frac of holes_tint inside windows = {m(arch['tint_in_win']):.3f}\n")
    for typ in ("tint", "struct"):
        print(f"{'TINT windows' if typ=='tint' else 'STRUCTURAL see-through holes (roof-rail/wheel/etc)'}:")
        print(f"  {'size':<7}{'GT#':>7}{'OURS det':>11}{'PROD det':>11}")
        tot = [0, 0, 0]
        for name, lo, hi in SIZES:
            g, o, p = counts[typ][name]
            for k in range(3): tot[k] += [g, o, p][k]
            orr = f"{o} ({100*o/g:.0f}%)" if g else "-"
            prr = f"{p} ({100*p/g:.0f}%)" if g else "-"
            print(f"  {name:<7}{g:>7}{orr:>11}{prr:>11}")
        orr = f"{tot[1]} ({100*tot[1]/tot[0]:.0f}%)" if tot[0] else "-"
        prr = f"{tot[2]} ({100*tot[2]/tot[0]:.0f}%)" if tot[0] else "-"
        print(f"  {'TOTAL':<7}{tot[0]:>7}{orr:>11}{prr:>11}\n")
    if ours_prec[1]:
        print(f"OUR silhouette structural-hole PRECISION: {ours_prec[0]}/{ours_prec[1]} = "
              f"{100*ours_prec[0]/ours_prec[1]:.0f}% of holes we punch are real (rest = spurious holes in the matte)")


if __name__ == "__main__":
    main()
