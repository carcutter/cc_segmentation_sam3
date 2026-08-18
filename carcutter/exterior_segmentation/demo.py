"""Run the pipeline over a folder of images and write masks + an RGBA cutout.

    export BIREFNET_REPO=/path/to/carcutter/BiRefNet
    python -m carcutter.exterior_segmentation.demo --images IN --out OUT --weights WEIGHTS_DIR
"""
import argparse
import glob
import os
import time

import numpy as np
from PIL import Image

from .pipeline import ExteriorSegmenter

LAYERS = ("outline", "punchout", "windows", "antenna")


def weight_paths(d):
    return {"detector": f"{d}/rfdetr_ca_unified_v1.pth",
            "outline": f"{d}/birefnet_aspect_prod_bucket5.pt",
            "trihead": f"{d}/birefnet_trihead_bucket.pt",
            "antenna_unet": f"{d}/unet_antenna_v3.pt"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", required=True, help="directory holding the four checkpoints")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=0)
    args = ap.parse_args()

    for d in LAYERS + ("cutout",):
        os.makedirs(f"{args.out}/{d}", exist_ok=True)
    seg = ExteriorSegmenter(weight_paths(args.weights), device=args.device)

    srcs = sorted(glob.glob(f"{args.images}/*.jpg") + glob.glob(f"{args.images}/*.png"))
    if args.n:
        srcs = srcs[:args.n]
    t0 = time.time()
    for i, src in enumerate(srcs):
        stem = os.path.splitext(os.path.basename(src))[0]
        rgb = np.array(Image.open(src).convert("RGB"))
        out = seg.predict(rgb)
        for k in LAYERS:
            Image.fromarray(out[k].astype(np.uint8) * 255).save(f"{args.out}/{k}/{stem}.png", optimize=True)
        rgba = np.dstack([rgb, seg.cutout_alpha(out).astype(np.uint8) * 255])
        Image.fromarray(rgba).save(f"{args.out}/cutout/{stem}.png")
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(srcs)}  {(time.time()-t0)/(i+1):.2f}s/img", flush=True)
    print(f"done: {len(srcs)} images in {(time.time()-t0)/60:.1f} min -> {args.out}")


if __name__ == "__main__":
    main()
