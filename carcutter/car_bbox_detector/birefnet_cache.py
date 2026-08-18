#!/usr/bin/env python3
"""
Cache a BiRefNet model's full-res masks over the e2e test set (detector-box crop,
same framing as training) so the final hybrid comparison needs no GPU.

Writes data/e2e/birefnet_<tag>/<stem>.npy (bool full-res mask).

Run from repo root:
  PYTHONPATH=.:carcutter/car_bbox_detector/birefnet/BiRefNet \
  python carcutter/car_bbox_detector/birefnet_cache.py --ckpt <...epoch_N.pth> --tag outline
"""
import argparse, csv
from pathlib import Path
import numpy as np, torch
from PIL import Image
from carcutter.car_bbox_detector.birefnet_eval import load_birefnet, birefnet_outline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True, help="outline | holes")
    ap.add_argument("--e2e", default="carcutter/car_bbox_detector/data/e2e")
    ap.add_argument("--index", default="carcutter/car_bbox_detector/data/index.csv")
    args = ap.parse_args()
    dev = "cuda"
    idx = {Path(r["image"]).stem: r for r in csv.DictReader(open(args.index))}
    model = load_birefnet(args.ckpt, dev)
    out = Path(args.e2e) / f"birefnet_{args.tag}"; out.mkdir(parents=True, exist_ok=True)
    npzs = sorted((Path(args.e2e) / "masks").glob("*.npz"))
    for i, npz in enumerate(npzs):
        r = idx.get(npz.stem)
        if r is None:
            continue
        d = np.load(npz)
        if "carbox" not in d:
            continue
        img = np.array(Image.open(r["image"]).convert("RGB"))
        m = birefnet_outline(model, img, d["carbox"], dev)
        np.save(out / f"{npz.stem}.npy", m)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(npzs)}", flush=True)
    print(f"cached -> {out}")


if __name__ == "__main__":
    main()
