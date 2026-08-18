#!/usr/bin/env python3
"""
Remove floating antenna false-positive blobs that don't connect to the car body.

Antennas attach to the body, so any antenna/outline component disconnected from the
main car blob (after bridging small gaps) is a false positive. This:
  - drops floating FP blobs from the OUTLINE (= body ∪ antenna) -> cleaner outline
  - drops floating FP blobs from the ANTENNA mask -> higher antenna IoU/precision
Bridges base->body gaps with a morphological close before the connectivity test, so
real-but-narrowly-attached antennas are kept.

Evaluates raw vs keep_main(close_px) on cached e2e masks (no GPU), vs GT.
Run from repo root:
  PYTHONPATH=. /home/rutger/miniconda3/envs/cc_sam3/bin/python carcutter/car_bbox_detector/antenna_fp_clean.py
"""
import argparse, csv
from pathlib import Path
import numpy as np, cv2
from PIL import Image
from carcutter.car_bbox_detector.seg_eval import boundary_f, mask_iou
BLUE, WHITE = (0, 0, 255), (255, 255, 255)


def keep_main(mask, close_px):
    """Keep only the pixels whose component (after a morphological close to bridge
    small gaps) belongs to the largest connected component (the car body)."""
    m = mask.astype(np.uint8)
    if m.sum() == 0:
        return mask
    if close_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
        bridged = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    else:
        bridged = m
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(bridged, 8)
    if n <= 1:
        return mask
    main = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return mask & (lbl == main)


def ncomp(m):
    n, _ = cv2.connectedComponents(m.astype(np.uint8)); return n - 1


def recall(g, p):
    d = g.sum(); return (g & p).sum() / d if d else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2e", default="carcutter/car_bbox_detector/data/e2e")
    ap.add_argument("--index", default="carcutter/car_bbox_detector/data/index.csv")
    args = ap.parse_args()
    idx = {Path(r["image"]).stem: r for r in csv.DictReader(open(args.index))}
    closes = [0, 5, 9, 15]

    out = {f"close{c}": {"o_iou": [], "o_bf1": [], "a_iou": [], "a_rec": [], "a_cc": []} for c in closes}
    out["raw"] = {"o_iou": [], "o_bf1": [], "a_iou": [], "a_rec": [], "a_cc": []}
    n_ant = 0
    for npz in sorted((Path(args.e2e) / "masks").glob("*.npz")):
        r = idx.get(npz.stem)
        if r is None:
            continue
        d = np.load(npz)
        rgb = np.array(Image.open(r["mask"]).convert("RGB"))
        gtu = rgb.sum(2) > 0; gtw = (rgb == WHITE).all(2)
        outline = d["outline"].astype(bool); antenna = d["antenna"].astype(bool)
        has_ant = gtw.sum() >= 20
        if has_ant:
            n_ant += 1
        # raw
        out["raw"]["o_iou"].append(mask_iou(gtu, outline))
        out["raw"]["o_bf1"].append(boundary_f(gtu, outline, 1))
        if has_ant:
            out["raw"]["a_iou"].append(mask_iou(gtw, antenna))
            out["raw"]["a_rec"].append(recall(gtw, antenna))
            out["raw"]["a_cc"].append(ncomp(antenna))
        for c in closes:
            o2 = keep_main(outline, c)
            a2 = antenna & o2          # antenna kept only where outline survives
            out[f"close{c}"]["o_iou"].append(mask_iou(gtu, o2))
            out[f"close{c}"]["o_bf1"].append(boundary_f(gtu, o2, 1))
            if has_ant:
                out[f"close{c}"]["a_iou"].append(mask_iou(gtw, a2))
                out[f"close{c}"]["a_rec"].append(recall(gtw, a2))
                out[f"close{c}"]["a_cc"].append(ncomp(a2))

    m = lambda a: float(np.mean(a)) if a else float("nan")
    print(f"Antenna FP-blob cleanup (keep main connected component), n_ant={n_ant}\n")
    print(f"  {'variant':<9}{'out_IoU':>9}{'out_BF1':>9}{'ant_IoU':>9}{'ant_rec':>9}{'ant_cc':>9}")
    for k in ["raw"] + [f"close{c}" for c in closes]:
        a = out[k]
        print(f"  {k:<9}" + "".join(f"{m(a[x]):>9.3f}" for x in ("o_iou", "o_bf1", "a_iou", "a_rec", "a_cc")))


if __name__ == "__main__":
    main()
