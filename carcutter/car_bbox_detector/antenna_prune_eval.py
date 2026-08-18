#!/usr/bin/env python3
"""
Antenna FP pruning via the placement-likelihood map — measure FP-drop vs recall.

For each predicted antenna connected component, score its LOCATION against the
per-view placement map (body frame from BiRefNet's body silhouette, view from
filename). Components whose best placement-likelihood < thr fall in a dead zone
(stick/branch/pole in an impossible spot) -> prune. Real roof/rack/mast antennas
touch a hotspot -> kept.

Reuses cached e2e_v2 masks (antenna = v1 UNet assembly) + cached BiRefNet body
(birefnet_outline/<stem>.npy for the body bbox). CPU only.

Reports baseline vs pruned@thr:
  - recall (antenna-present imgs)      [higher better, must hold]
  - fire_noAnt% (no-antenna imgs)      [lower better — the FP we want gone]
  - FP_cc (FP components / antenna img) [lower better]

Run from repo root:
  PYTHONPATH=. python carcutter/car_bbox_detector/antenna_prune_eval.py
"""
import argparse, csv, re
from pathlib import Path
import numpy as np, cv2
from PIL import Image
WHITE = (255, 255, 255)
THRS = [0.0, 0.01, 0.03, 0.05, 0.10, 0.20]


def view_of(p):
    m = re.search(r'-(front-left|front-right|rear-left|rear-right|side-left|side-right|front|rear|side)\.jpg$', p)
    return m.group(1) if m else "other"


def comp_scores(mask, body_bbox, vmap, extent):
    """Per connected component: (component_mask, best placement-likelihood)."""
    U0, U1, V0, V1, GW, GH = extent
    bx0, by0, bw, bh = body_bbox
    n, lbl = cv2.connectedComponents(mask.astype(np.uint8))
    out = []
    for i in range(1, n):
        comp = lbl == i
        ys, xs = np.where(comp)
        u = (xs - bx0) / bw; v = (ys - by0) / bh
        gi = ((u - U0) / (U1 - U0) * GW).astype(int)
        gj = ((v - V0) / (V1 - V0) * GH).astype(int)
        ok = (gi >= 0) & (gi < int(GW)) & (gj >= 0) & (gj < int(GH))
        vals = vmap[gj[ok], gi[ok]] if ok.any() else np.array([0.0])
        score = float(vals.mean()) if SCORE_STAT == "mean" else float(vals.max())
        out.append((comp, score))
    return out


SCORE_STAT = "max"


def prune(mask, body_bbox, vmap, extent, thr):
    out = np.zeros_like(mask)
    for comp, score in comp_scores(mask, body_bbox, vmap, extent):
        if score >= thr:
            out |= comp
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2e", default="carcutter/car_bbox_detector/data/e2e_v2")
    ap.add_argument("--index", default="carcutter/car_bbox_detector/data/index.csv")
    ap.add_argument("--maps", default="carcutter/car_bbox_detector/data/antenna_placement.npz")
    ap.add_argument("--bf-body", default="carcutter/car_bbox_detector/data/e2e/birefnet_outline",
                    help="dir of cached BiRefNet body masks (<stem>.npy) for the body bbox")
    ap.add_argument("--stat", default="max", choices=["max", "mean"])
    ap.add_argument("--body-source", default="birefnet", choices=["birefnet", "gt"])
    args = ap.parse_args()
    global SCORE_STAT; SCORE_STAT = args.stat
    idx = {Path(r["image"]).stem: r for r in csv.DictReader(open(args.index))}
    z = np.load(args.maps); extent = z["extent"]
    vmaps = {k[5:]: z[k] for k in z.files if k.startswith("map__")}
    bf_body = Path(args.bf_body)

    agg = {t: {"rec": [], "fire": [], "fpcc": []} for t in THRS}
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    n_used = 0
    for npz in sorted((Path(args.e2e) / "masks").glob("*.npz")):
        r = idx.get(npz.stem)
        if r is None:
            continue
        rgb = np.array(Image.open(r["mask"]).convert("RGB"))
        white = (rgb == WHITE).all(2)
        if args.body_source == "gt":
            body = (rgb.sum(2) > 0) & (~white)
        else:
            bfp = bf_body / f"{npz.stem}.npy"
            if not bfp.exists():
                continue
            body = np.load(bfp)
        ys, xs = np.where(body)
        if len(xs) < 50:
            continue
        bbox = (xs.min(), ys.min(), max(1, xs.max() - xs.min()), max(1, ys.max() - ys.min()))
        ant = np.load(npz)["antenna"].astype(bool)
        has_ant = white.sum() >= 20
        wd = cv2.dilate(white.astype(np.uint8), k) > 0
        view = view_of(r["image"])
        vmap = vmaps.get(view, vmaps["all"])
        n_used += 1
        for t in THRS:
            pm = ant if t == 0.0 else prune(ant, bbox, vmap, extent, t)
            if has_ant:
                agg[t]["rec"].append((white & pm).sum() / white.sum())
                nn, ll = cv2.connectedComponents(pm.astype(np.uint8)); fp = 0
                for ci in range(1, nn):
                    c = ll == ci
                    if (c & wd).sum() < 0.15 * c.sum():
                        fp += 1
                agg[t]["fpcc"].append(fp)
            else:
                agg[t]["fire"].append(1.0 if pm.sum() >= 20 else 0.0)

    m = lambda a: float(np.mean(a)) if a else float("nan")
    print(f"Antenna placement-prune sweep, n={n_used} "
          f"({len(agg[0.0]['rec'])} antenna imgs, {len(agg[0.0]['fire'])} no-antenna imgs)\n")
    print(f"  {'thr':>6}{'recall':>9}{'fire_noAnt%':>13}{'FP_cc':>9}")
    for t in THRS:
        a = agg[t]
        print(f"  {t:>6.2f}{m(a['rec']):>9.3f}{100*m(a['fire']):>12.1f}%{m(a['fpcc']):>9.2f}")


if __name__ == "__main__":
    main()
