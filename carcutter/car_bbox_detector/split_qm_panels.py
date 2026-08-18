#!/usr/bin/env python3
"""
Split the quality-manager Exterior set into its three panels.

Each delivered JPG is a horizontal 3-panel comparison strip:
  panel 1 = original dealer photo   -> input/   (the input to our pipeline)
  panel 2 = current production cutout, composited on the studio background -> prod/
  panel 3 = QC-retouched cutout, same background/framing as panel 2        -> qc/

Panels 2 and 3 share framing and background (they differ only where the two cutouts
disagree); panel 1 is a different framing entirely (prod re-crops onto the new plate).

Run: python carcutter/car_bbox_detector/split_qm_panels.py
"""
import glob
import os
from multiprocessing import Pool

from PIL import Image

ROOT = "/home/rutger/work/cc_segmentation_sam3/data/car_segmentation/qm_eval_202608_car_segmentation"
NAMES = ["input", "prod", "qc"]


def work(src):
    stem = os.path.splitext(os.path.basename(src))[0]
    im = Image.open(src)
    W, H = im.size
    # 6 of 490 strips have a width not divisible by 3 (exporter rounding); round the
    # boundaries instead of flooring so the last panel keeps its full width.
    edges = [round(i * W / 3) for i in range(4)]
    if not 1.1 < (W / 3) / H < 2.0:
        return ("badaspect", stem)
    for i, name in enumerate(NAMES):
        im.crop((edges[i], 0, edges[i + 1], H)).save(f"{ROOT}/{name}/{stem}.jpg", quality=95)
    return ("ok", stem)


def main():
    for n in NAMES:
        os.makedirs(f"{ROOT}/{n}", exist_ok=True)
    srcs = sorted(glob.glob(f"{ROOT}/raw/*.jpg"))
    with Pool(16) as p:
        res = p.map(work, srcs, chunksize=8)
    ok = [s for st, s in res if st == "ok"]
    bad = [s for st, s in res if st != "ok"]
    print(f"split {len(ok)} strips -> {'/'.join(NAMES)}")
    for s in bad:
        print("  SKIPPED (panel aspect out of range):", s)


if __name__ == "__main__":
    main()
