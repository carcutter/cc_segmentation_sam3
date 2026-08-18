#!/usr/bin/env python3
"""
Export the quality-manager Exterior results as multi-layer Photoshop PSDs for QC validation.

Consumes the masks already written by run_qm_eval.py (no GPU, re-runnable) and composes one PSD
per image on the panel-1 input canvas.

Layer stack (Photoshop panel, top -> bottom):
  antenna_ADDED         hidden   white fill, alpha = pixels the antenna model added to the body
  windows_tint          hidden   white fill, alpha = tint/see-through glass
  punchout_holes        hidden   white fill, alpha = structural see-through (wheel gaps, rails, ...)
  car_outline           hidden   white fill, alpha = our silhouette  <- the layer to judge
  contour [ours]        VISIBLE  thin stroke along our cut line
  cutout [ours]         VISIBLE  the extraction, as a plain RGBA layer
  ORIGINAL (uncropped)  hidden   the untouched panel-1 photo, full frame

QA WORKFLOW — OVER-CROP CHECK: toggle `ORIGINAL (uncropped)` on. It is the full photo with nothing
removed, and `contour [ours]` draws our cut line straight onto it, so anything we sliced off the
vehicle is immediately visible against the real pixels. (A layer mask alone was not enough here: it
hides the discarded pixels, and recovering them means knowing to shift-click the mask thumbnail.)

QA WORKFLOW FOR A BAD ANTENNA: the antenna is isolated in `antenna_ADDED` — exactly the pixels the
zoom UNet contributed on top of the bare BiRefNet body, nothing else. If it caught a roof rail or a
background pole instead of a real antenna, Ctrl/Cmd-click that layer to load it as a selection, then
delete on the `cutout [ours]` layer to drop it. The rest of the cutout is untouched, because the body
silhouette never depended on that model. Measured rate of a genuinely wrong antenna piece:
~0.4% of images (1 in 247), with a further ~8% carrying a <=10px sliver of boundary slop on the real
antenna that is cosmetic rather than wrong.

Mask layers follow the repo convention from carcutter/inference/run_psd_export.py: white RGBA with
the binary mask as alpha, so Ctrl/Cmd-clicking the layer thumbnail loads it as a selection.
They ship hidden so the PSD opens on the extraction; QC toggles what they want to inspect.

Run: python carcutter/car_bbox_detector/export_qm_psd.py [--n 6] [--with-refs]
"""
import argparse
import glob
import os
import struct

import cv2
import numpy as np
from PIL import Image
from psd_tools import PSDImage
from psd_tools.constants import BlendMode, Compression, Resource
from psd_tools.psd.image_resources import ImageResource

# RLE, not ZIP. Photoshop refuses to open these files when the channels or (especially) the merged
# image data are ZIP-compressed — it always writes RLE itself, and its reader is strict where GIMP and
# ImageMagick are lenient. The repo's production trailer exporter (carcutter/inference/run_psd_export.py)
# uses psd-tools' RLE defaults and opens fine, so we match it. Costs ~15% size; non-negotiable.
COMP = Compression.RLE

# 72 dpi ResolutionInfo (resource 1005). psd-tools omits it; Photoshop expects it on a normal file.
RESOLUTION_INFO = struct.pack(">IHHIHH", 72 << 16, 1, 1, 72 << 16, 1, 1)

SET = "/home/rutger/work/cc_segmentation_sam3/data/car_segmentation/qm_eval_202608_car_segmentation"
MASK_OPACITY = 204  # 80% — same as the trailer PSD exporter
# (directory, layer name) in Photoshop bottom -> top order above the cutout
MASK_LAYERS = [("outline", "car_outline"), ("punchout", "punchout_holes"),
               ("windows", "windows_tint"), ("antenna", "antenna_ADDED")]


def mask_to_rgba(mask):
    """White fill, alpha = mask — loads as a Photoshop selection."""
    rgba = np.zeros((*mask.shape, 4), np.uint8)
    rgba[..., :3] = 255
    rgba[..., 3] = mask
    return Image.fromarray(rgba, mode="RGBA")


def cutout_rgba(img, alpha):
    rgba = np.zeros((*alpha.shape, 4), np.uint8)
    rgba[..., :3] = img
    rgba[..., 3] = alpha
    return Image.fromarray(rgba, mode="RGBA")


def contour_rgba(mask, W, H, color=(0, 255, 0)):
    """Thin stroke along the cut line — cheap (mostly transparent) and readable over the raw photo."""
    m = mask.astype(np.uint8)
    stroke = max(2, round(min(W, H) / 500))
    edge = cv2.dilate(m, np.ones((stroke, stroke), np.uint8)) - cv2.erode(m, np.ones((stroke, stroke), np.uint8))
    rgba = np.zeros((H, W, 4), np.uint8)
    rgba[..., :3] = color
    rgba[..., 3] = (edge > 0).astype(np.uint8) * 255
    return Image.fromarray(rgba, mode="RGBA")


def load(path):
    return np.array(Image.open(path).convert("L")) if os.path.exists(path) else None


def compose(stem, src, mask_dir, out_path, with_refs=False):
    pil = Image.open(src).convert("RGB")
    W, H = pil.size
    masks = {name: load(f"{mask_dir}/{d}/{stem}.png") for d, name in MASK_LAYERS}
    if masks["car_outline"] is None:
        return None, "no outline mask (car not detected)"

    psd = PSDImage.new("RGB", (W, H))
    n = 0

    alpha = masks["car_outline"].copy()
    if masks["punchout_holes"] is not None:
        alpha[masks["punchout_holes"] > 0] = 0  # carve see-through gaps out of the extraction

    # psd-tools appends bottom -> top.
    # Full uncropped frame at the bottom, hidden: the over-crop reference. Costs a second copy of the
    # photo pixels (~6 MB) — bought deliberately, a layer mask alone hides what was cut away.
    psd.create_pixel_layer(pil, name="ORIGINAL (uncropped)", opacity=255,
                           blend_mode=BlendMode.NORMAL, compression=COMP).visible = False
    n += 1

    # A plain RGBA layer, NOT a layer mask: psd-tools' create_mask() is the other thing Photoshop
    # would not accept. Same result on screen, and erasing here is just as easy for QA.
    psd.create_pixel_layer(cutout_rgba(np.array(pil), alpha), name="cutout [ours]",
                           opacity=255, blend_mode=BlendMode.NORMAL, compression=COMP)
    n += 1

    psd.create_pixel_layer(contour_rgba(masks["car_outline"] > 127, W, H), name="contour [ours]",
                           opacity=255, compression=COMP)
    n += 1

    if with_refs:
        for tag, dname in (("reference: current prod", "prod"), ("reference: QC retouched", "qc")):
            p = f"{SET}/{dname}/{stem}.jpg"
            if os.path.exists(p):
                # Different framing from panel 1 (prod re-crops onto its plate) — context only.
                ref = Image.open(p).convert("RGB").resize((W, H))
                psd.create_pixel_layer(ref, name=tag, opacity=255, compression=COMP).visible = False
                n += 1

    for _d, name in MASK_LAYERS:
        m = masks[name]
        if m is None or m.max() == 0:
            continue  # skip empty layers (e.g. no antenna on this vehicle)
        psd.create_pixel_layer(mask_to_rgba(m), name=name, opacity=MASK_OPACITY,
                               compression=COMP).visible = False
        n += 1

    # psd-tools stores the flattened preview RAW (~9 MB for a 2000x1500 panel); re-encode it RLE,
    # which is what Photoshop itself writes here.
    flat = np.array(psd.composite().convert("RGB"))
    psd._record.image_data.compression = COMP
    psd._record.image_data.set_data([flat[..., c].tobytes() for c in range(3)], psd._record.header)
    psd._record.image_resources[Resource.RESOLUTION_INFO] = ImageResource(
        key=Resource.RESOLUTION_INFO, data=RESOLUTION_INFO)

    psd.save(out_path)
    return n, None


def verify(out_path, n_expected):
    if os.path.getsize(out_path) == 0:
        return "file is empty"
    try:
        psd = PSDImage.open(out_path)
    except Exception as exc:
        return f"cannot reopen: {exc}"
    got = len(list(psd))
    return None if got == n_expected else f"layer count {got} != {n_expected}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="limit images (0 = all)")
    ap.add_argument("--masks", default=f"{SET}/ours")
    ap.add_argument("--out", default=f"{SET}/psd")
    ap.add_argument("--with-refs", action="store_true",
                    help="include the prod / QC-retouched panels as hidden reference layers")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    srcs = sorted(glob.glob(f"{SET}/input/*.jpg"))
    if args.n:
        srcs = srcs[:args.n]

    ok, skipped, bad, sizes = 0, [], [], []
    for src in srcs:
        stem = os.path.splitext(os.path.basename(src))[0]
        out_path = f"{args.out}/{stem}.psd"
        n, err = compose(stem, src, args.masks, out_path, args.with_refs)
        if err:
            skipped.append((stem, err))
            continue
        v = verify(out_path, n)
        if v:
            bad.append((stem, v))
        else:
            ok += 1
            sizes.append(os.path.getsize(out_path) / 1e6)

    print(f"\nPSD export: {ok}/{len(srcs)} OK -> {args.out}")
    if sizes:
        print(f"  size {min(sizes):.1f}-{max(sizes):.1f} MB (avg {sum(sizes)/len(sizes):.1f}), "
              f"total {sum(sizes)/1000:.2f} GB")
    for stem, why in skipped:
        print(f"  SKIPPED {stem}: {why}")
    for stem, why in bad:
        print(f"  FAILED  {stem}: {why}")


if __name__ == "__main__":
    main()
