#!/usr/bin/env python3
"""
Chain zoom-in evaluation.

Compares two chain segmentation strategies on the same images:

  Pass 1 – image-level:  standard SAM3 "chain" inference on the full image.
            The whole 1008×1008 model budget is spent on the full image, so
            individual chain links are blurry / merged blobs.

  Pass 2 – zoom-in:      for each instance bounding box from pass 1, crop a
            padded region and re-run "chain" inference at that crop's native
            resolution.  Because SAM3 still uses its full 1008×1008 budget,
            every pixel of the crop gets ~N× more attention than it did at
            image level.  The per-crop masks are back-projected to the
            original coordinate space and OR-combined.

Output per image:
  • <stem>_comparison.jpg   – left: image-level mask overlay
                               right: zoom-in mask overlay
  • <stem>_crops/           – per-instance 2-panel (low-res | high-res crop)
  summary.txt               – pixel-coverage and IoU statistics

Usage:
    python carcutter/eval/analyze_chain_zoom.py \\
        --input-dir  /path/to/images \\
        --output-dir /path/to/eval_output \\
        [--prompt "chain"] \\
        [--padding 0.4] \\
        [--threshold 0.3] \\
        [--device cuda]
"""

import argparse
import os
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from sam3 import build_sam3_image_model
from sam3.eval.postprocessors import PostProcessImage
from sam3.model.utils.misc import copy_data_to_device
from sam3.train.data.collator import collate_fn_api as collate
from sam3.train.data.sam3_image_dataset import (
    Datapoint,
    FindQueryLoaded,
    Image as SAMImage,
    InferenceMetadata,
)
from sam3.train.transforms.basic_for_api import (
    ComposeAPI,
    NormalizeAPI,
    RandomResizeAPI,
    ToTensorAPI,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# Overlay colors (RGB)
COLOR_LOWRES  = (30,  144, 255)   # dodger blue
COLOR_HIGHRES = (50,  205,  50)   # lime green
ALPHA = 0.45


# ---------------------------------------------------------------------------
# SAM3 inference helpers
# ---------------------------------------------------------------------------

def _build_transform() -> ComposeAPI:
    return ComposeAPI(
        transforms=[
            RandomResizeAPI(sizes=1008, max_size=1008, square=True, consistent_transform=False),
            ToTensorAPI(),
            NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def _make_postprocessor(threshold: float) -> PostProcessImage:
    return PostProcessImage(
        max_dets_per_img=-1,
        iou_type="segm",
        use_original_sizes_box=True,
        use_original_sizes_mask=True,
        convert_mask_to_rle=False,
        detection_threshold=threshold,
        to_cpu=False,
    )


def _create_datapoint(pil_image: Image.Image, prompt: str, query_id: int) -> Datapoint:
    w, h = pil_image.size
    dp = Datapoint(find_queries=[], images=[])
    dp.images = [SAMImage(data=pil_image, objects=[], size=[h, w])]
    dp.find_queries.append(
        FindQueryLoaded(
            query_text=prompt,
            image_id=0,
            object_ids_output=[],
            is_exhaustive=True,
            query_processing_order=0,
            inference_metadata=InferenceMetadata(
                coco_image_id=query_id,
                original_image_id=query_id,
                original_category_id=1,
                original_size=[h, w],
                object_id=0,
                frame_index=0,
            ),
        )
    )
    return dp


def _infer(model, postprocessor, transform, pil_image: Image.Image,
           prompt: str, query_id: int, device: str) -> dict | None:
    """Run one forward pass; return raw result dict or None if no detection."""
    dp = _create_datapoint(pil_image, prompt, query_id)
    dp = transform(dp)
    batch = collate([dp], dict_key="inference")["inference"]
    batch = copy_data_to_device(batch, torch.device(device), non_blocking=True)
    output = model(batch)
    processed = postprocessor.process_results(output, batch.find_metadatas)
    return processed.get(query_id)


def _masks_tensor_to_np(masks_tensor: torch.Tensor) -> np.ndarray:
    """[N, (1,) H, W] bool tensor → [N, H, W] uint8 numpy (0/255)."""
    if masks_tensor.dim() == 4:
        masks_tensor = masks_tensor.squeeze(1)
    return (masks_tensor.cpu().numpy() > 0).astype(np.uint8) * 255


def _flatten(masks_np: np.ndarray) -> np.ndarray:
    """[N, H, W] uint8 → [H, W] uint8 via OR."""
    return masks_np.any(axis=0).astype(np.uint8) * 255


# ---------------------------------------------------------------------------
# Crop / back-projection helpers
# ---------------------------------------------------------------------------

def _padded_crop(pil_image: Image.Image, box_xyxy, padding_frac: float
                 ) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Crop image around a bbox with proportional padding.

    Returns (cropped_image, (cx1, cy1, cx2, cy2)) in original pixel coords.
    """
    W, H = pil_image.size
    x1, y1, x2, y2 = (float(v) for v in box_xyxy)
    bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)
    pad_x = bw * padding_frac
    pad_y = bh * padding_frac
    cx1 = max(0,  int(x1 - pad_x))
    cy1 = max(0,  int(y1 - pad_y))
    cx2 = min(W,  int(x2 + pad_x))
    cy2 = min(H,  int(y2 + pad_y))
    # guard against degenerate crops
    if cx2 <= cx1:
        cx2 = min(W, cx1 + 1)
    if cy2 <= cy1:
        cy2 = min(H, cy1 + 1)
    return pil_image.crop((cx1, cy1, cx2, cy2)), (cx1, cy1, cx2, cy2)


def _backproject(crop_mask_np: np.ndarray, crop_coords: tuple, orig_wh: tuple
                 ) -> np.ndarray:
    """Paste a (crop_h, crop_w) mask back into a full-image (H, W) mask."""
    W, H = orig_wh
    cx1, cy1, cx2, cy2 = crop_coords
    expected_h, expected_w = cy2 - cy1, cx2 - cx1
    if crop_mask_np.shape != (expected_h, expected_w):
        crop_mask_np = np.array(
            Image.fromarray(crop_mask_np).resize(
                (expected_w, expected_h), Image.NEAREST
            )
        )
    full = np.zeros((H, W), dtype=np.uint8)
    full[cy1:cy2, cx1:cx2] = crop_mask_np
    return full


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def _overlay(pil_image: Image.Image, mask_np: np.ndarray,
             color: tuple, alpha: float) -> np.ndarray:
    """Return RGB numpy array with mask overlaid in color."""
    arr = np.array(pil_image.convert("RGB"), dtype=np.float32)
    m = mask_np > 0
    for c, col in enumerate(color):
        arr[:, :, c] = np.where(m, arr[:, :, c] * (1 - alpha) + col * alpha, arr[:, :, c])
    return np.clip(arr, 0, 255).astype(np.uint8)


def _save_comparison(pil_image: Image.Image,
                     lowres_mask: np.ndarray,
                     highres_mask: np.ndarray,
                     output_path: str,
                     title: str,
                     lowres_n_instances: int,
                     highres_coverage_pct: float,
                     lowres_coverage_pct: float,
                     iou: float) -> None:
    """Save a 3-panel figure: original | low-res overlay | high-res overlay."""
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    fig.suptitle(title, fontsize=11, y=1.01)

    # Panel 1: original
    axes[0].imshow(pil_image)
    axes[0].set_title("Original", fontsize=10)

    # Panel 2: image-level (low-res)
    axes[1].imshow(_overlay(pil_image, lowres_mask, COLOR_LOWRES, ALPHA))
    patch_lr = mpatches.Patch(color=[c/255 for c in COLOR_LOWRES],
                               label=f"image-level  coverage={lowres_coverage_pct:.1f}%  "
                                     f"({lowres_n_instances} instance(s))")
    axes[1].legend(handles=[patch_lr], loc="lower left", fontsize=8,
                   framealpha=0.7)
    axes[1].set_title("Image-level mask", fontsize=10)

    # Panel 3: zoom-in (high-res)
    axes[2].imshow(_overlay(pil_image, highres_mask, COLOR_HIGHRES, ALPHA))
    patch_hr = mpatches.Patch(color=[c/255 for c in COLOR_HIGHRES],
                               label=f"zoom-in  coverage={highres_coverage_pct:.1f}%  "
                                     f"IoU vs low-res={iou:.3f}")
    axes[2].legend(handles=[patch_hr], loc="lower left", fontsize=8,
                   framealpha=0.7)
    axes[2].set_title("Zoom-in mask (back-projected)", fontsize=10)

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _save_crop_comparison(pil_crop: Image.Image,
                          lowres_crop_mask: np.ndarray,
                          highres_crop_mask: np.ndarray,
                          output_path: str,
                          instance_idx: int,
                          box_xyxy) -> None:
    """Save a 2-panel crop-level comparison for one detected instance."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Instance {instance_idx}  box=({int(box_xyxy[0])},{int(box_xyxy[1])})–"
        f"({int(box_xyxy[2])},{int(box_xyxy[3])})",
        fontsize=10
    )

    axes[0].imshow(_overlay(pil_crop, lowres_crop_mask, COLOR_LOWRES,  ALPHA))
    axes[0].set_title("Image-level (crop region)", fontsize=9)

    axes[1].imshow(_overlay(pil_crop, highres_crop_mask, COLOR_HIGHRES, ALPHA))
    axes[1].set_title("Zoom-in inference (same crop region)", fontsize=9)

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-image evaluation
# ---------------------------------------------------------------------------

def _compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a > 0
    b = mask_b > 0
    inter = (a & b).sum()
    union = (a | b).sum()
    return float(inter) / float(union) if union > 0 else 1.0


def _coverage_pct(mask: np.ndarray) -> float:
    return 100.0 * (mask > 0).sum() / mask.size


def evaluate_image(
    model,
    postprocessor,
    transform,
    pil_image: Image.Image,
    prompt: str,
    padding_frac: float,
    device: str,
    img_idx: int,
    output_dir: str,
    stem: str,
) -> dict:
    """Run both passes on one image; save visualisations; return stats dict."""
    W, H = pil_image.size

    # ── Pass 1: image-level ─────────────────────────────────────────────────
    result_full = _infer(model, postprocessor, transform, pil_image,
                         prompt, query_id=img_idx * 10000, device=device)

    no_chain = (
        result_full is None
        or "masks" not in result_full
        or len(result_full["masks"]) == 0
    )
    if no_chain:
        return {"stem": stem, "no_detection": True}

    masks_np    = _masks_tensor_to_np(result_full["masks"])   # [N, H, W]
    boxes       = result_full["boxes"].cpu()                   # [N, 4] XYXY
    n_instances = len(masks_np)
    lowres_mask = _flatten(masks_np)                           # [H, W] full-image

    # ── Pass 2: zoom-in per instance ─────────────────────────────────────────
    highres_full = np.zeros((H, W), dtype=np.uint8)
    crops_dir = os.path.join(output_dir, f"{stem}_crops")
    os.makedirs(crops_dir, exist_ok=True)

    for inst_idx, (inst_mask_np, box) in enumerate(zip(masks_np, boxes)):
        pil_crop, crop_coords = _padded_crop(pil_image, box, padding_frac)
        cx1, cy1, cx2, cy2 = crop_coords

        # Run zoom-in inference on the crop
        qid = img_idx * 10000 + inst_idx + 1
        result_crop = _infer(model, postprocessor, transform, pil_crop,
                             prompt, query_id=qid, device=device)

        has_zoom = (
            result_crop is not None
            and "masks" in result_crop
            and len(result_crop["masks"]) > 0
        )
        if has_zoom:
            zoom_masks_np = _masks_tensor_to_np(result_crop["masks"])
            zoom_flat     = _flatten(zoom_masks_np)                 # at crop res
            zoomed_full   = _backproject(zoom_flat, crop_coords, (W, H))
            highres_full  = np.maximum(highres_full, zoomed_full)
        else:
            zoom_flat   = np.zeros((cy2 - cy1, cx2 - cx1), dtype=np.uint8)
            zoomed_full = np.zeros((H, W), dtype=np.uint8)

        # Low-res crop: extract the corresponding region from the image-level mask
        lowres_crop = inst_mask_np[cy1:cy2, cx1:cx2]

        # Save crop-level comparison
        crop_path = os.path.join(crops_dir, f"instance_{inst_idx:02d}.jpg")
        _save_crop_comparison(
            pil_crop, lowres_crop, zoom_flat,
            crop_path, inst_idx, box
        )

    # ── Compute stats ────────────────────────────────────────────────────────
    iou = _compute_iou(lowres_mask, highres_full)
    lr_cov  = _coverage_pct(lowres_mask)
    hr_cov  = _coverage_pct(highres_full)

    # ── Save full-image comparison ───────────────────────────────────────────
    comp_path = os.path.join(output_dir, f"{stem}_comparison.jpg")
    _save_comparison(
        pil_image, lowres_mask, highres_full, comp_path,
        title=f"{stem}  ({n_instances} instance(s) detected)",
        lowres_n_instances=n_instances,
        lowres_coverage_pct=lr_cov,
        highres_coverage_pct=hr_cov,
        iou=iou,
    )

    return {
        "stem": stem,
        "no_detection": False,
        "n_instances": n_instances,
        "lowres_coverage_pct": lr_cov,
        "highres_coverage_pct": hr_cov,
        "iou": iou,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_chain_zoom_eval(
    input_dir: str,
    output_dir: str,
    prompt: str = "chain",
    padding_frac: float = 0.4,
    threshold: float = 0.3,
    device: str = "cuda",
) -> None:
    image_files = sorted(
        p for p in Path(input_dir).iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_files:
        print(f"No images found in {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    print(f"Found {len(image_files)} images")
    print(f"Prompt    : '{prompt}'")
    print(f"Padding   : {padding_frac:.0%}")
    print(f"Threshold : {threshold}")
    print(f"Device    : {device}")
    print(f"Output    : {output_dir}")
    print("\nLoading model...")

    model = build_sam3_image_model(device=device, eval_mode=True, load_from_HF=True)
    transform     = _build_transform()
    postprocessor = _make_postprocessor(threshold)

    rows = []
    autocast_device = "cuda" if device.startswith("cuda") else "cpu"
    with torch.inference_mode(), torch.autocast(autocast_device, dtype=torch.bfloat16):
        for img_idx, img_path in enumerate(tqdm(image_files, desc="Evaluating")):
            pil_image = Image.open(img_path).convert("RGB")
            stats = evaluate_image(
                model, postprocessor, transform, pil_image,
                prompt=prompt,
                padding_frac=padding_frac,
                device=device,
                img_idx=img_idx,
                output_dir=output_dir,
                stem=img_path.stem,
            )
            rows.append(stats)
            if stats["no_detection"]:
                tqdm.write(f"  {img_path.name}: no {prompt} detected, skipped")

    _write_summary(rows, output_dir, prompt, padding_frac, threshold)


def _write_summary(rows: list[dict], output_dir: str, prompt: str,
                   padding_frac: float, threshold: float) -> None:
    detected = [r for r in rows if not r["no_detection"]]
    skipped  = [r for r in rows if r["no_detection"]]

    lines = [
        f"Chain zoom-in evaluation",
        f"{'=' * 60}",
        f"Prompt        : '{prompt}'",
        f"Padding       : {padding_frac:.0%}",
        f"Threshold     : {threshold}",
        f"Total images  : {len(rows)}",
        f"  With chain  : {len(detected)}",
        f"  No chain    : {len(skipped)}",
    ]

    if detected:
        ious      = [r["iou"]                  for r in detected]
        lr_covs   = [r["lowres_coverage_pct"]  for r in detected]
        hr_covs   = [r["highres_coverage_pct"] for r in detected]
        n_insts   = [r["n_instances"]          for r in detected]

        lines += [
            "",
            "── Per-image averages (detected images only) ──────────────",
            f"  IoU low-res vs high-res : mean={np.mean(ious):.3f}  "
            f"median={np.median(ious):.3f}  "
            f"min={np.min(ious):.3f}  max={np.max(ious):.3f}",
            f"  Low-res  coverage %     : mean={np.mean(lr_covs):.2f}%  "
            f"median={np.median(lr_covs):.2f}%",
            f"  High-res coverage %     : mean={np.mean(hr_covs):.2f}%  "
            f"median={np.median(hr_covs):.2f}%",
            f"  Instances per image     : mean={np.mean(n_insts):.1f}  "
            f"max={np.max(n_insts)}",
            "",
            "── Interpretation ─────────────────────────────────────────",
            textwrap.fill(
                "IoU close to 1.0 → zoom-in and image-level masks agree well "
                "(no extra detail recovered). IoU much lower → zoom-in finds "
                "significantly different / more granular structure. "
                "High-res coverage > low-res coverage → zoom-in expands the "
                "detected region; < → it is more conservative.",
                width=60, initial_indent="  ", subsequent_indent="  "
            ),
        ]

        lines += ["", "── Per-image results ──────────────────────────────────────"]
        lines += [f"  {'Image':<50} {'n':>3}  {'IoU':>6}  {'LR%':>6}  {'HR%':>6}"]
        lines += [f"  {'-'*50} {'-'*3}  {'-'*6}  {'-'*6}  {'-'*6}"]
        for r in sorted(detected, key=lambda x: x["iou"]):
            lines.append(
                f"  {r['stem'][:50]:<50} {r['n_instances']:>3}  "
                f"{r['iou']:>6.3f}  {r['lowres_coverage_pct']:>6.2f}  "
                f"{r['highres_coverage_pct']:>6.2f}"
            )

    summary_path = os.path.join(output_dir, "summary.txt")
    text = "\n".join(lines) + "\n"
    with open(summary_path, "w") as f:
        f.write(text)
    print(f"\n{text}")
    print(f"Summary: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare image-level vs zoom-in chain segmentation. "
            "For each detected chain instance, crops a padded region and "
            "re-runs inference at that crop's full 1008×1008 resolution."
        )
    )
    parser.add_argument("--input-dir",  required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt",    default="chain",
                        help="Text prompt to evaluate (default: 'chain')")
    parser.add_argument("--padding",   type=float, default=0.4,
                        help="Padding as fraction of bbox size (default: 0.4 = 40%%)")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Detection threshold (default: 0.3)")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_chain_zoom_eval(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        prompt=args.prompt,
        padding_frac=args.padding,
        threshold=args.threshold,
        device=args.device,
    )


if __name__ == "__main__":
    main()
