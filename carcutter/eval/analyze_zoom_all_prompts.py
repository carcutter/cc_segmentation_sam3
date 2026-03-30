#!/usr/bin/env python3
"""
Zoom-in analysis across all prompts.

Evaluates how much zoom-in inference improves segmentation granularity for
objects of varying sizes: chains, wheels, axles, frames, shadows, etc.

For each image:
  Pass 1 – multi-prompt image-level:  all prompts packed into one forward
            pass per image.  Gives per-instance masks + boxes at full-image
            resolution for every prompt simultaneously.

  Pass 2 – zoom-in per merged crop:   instead of cropping each detected
            instance individually, we first pad each instance bbox, then
            merge any overlapping padded regions into a single crop.  This
            keeps nearby instances (e.g. adjacent chain links) in context.
            A minimum crop size prevents crops so tiny the model loses all
            context.  Each merged crop is inferred once; masks are
            back-projected and OR-combined per prompt.

Key metrics:
  IoU        – agreement between image-level and zoom-in masks.
               Lower → zoom-in finds meaningfully different structure.
  retention  – zoom-in coverage / image-level coverage.
               < 1 → zoom-in is more conservative (or object disappeared).
               > 1 → zoom-in expands the mask.
  disap%     – fraction of detected images where retention < 0.2,
               i.e. the zoom-in mask has essentially vanished.
               High disap% means zoom-in is harmful for this prompt.

Output:
  summary.txt               – per-prompt stats table + interpretation
  zoom_impact_chart.jpg     – chart: IoU, retention, disappearance by prompt
  <stem>_<prompt>_comp.jpg  – full-image 3-panel comparison
                              (only when --save-comparisons is set)

Usage:
    python carcutter/eval/analyze_zoom_all_prompts.py \\
        --input-dir  /path/to/images \\
        --output-dir /path/to/eval_output \\
        [--prompts "chain,cable,..."]  # default: all 16 prompts
        [--padding  0.4] \\
        [--min-crop-px 300] \\
        [--disappearance-threshold 0.2] \\
        [--threshold 0.3] \\
        [--save-comparisons] \\
        [--max-images 50] \\
        [--device cuda]
"""

import argparse
import os
import re
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


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_PROMPTS = [
    "chain",
    "safety chain",
    "cable",
    "wheel",
    "tire",
    "axle",
    "hitch",
    "coupler",
    "trailer tongue",
    "ramp",
    "gate",
    "trailer frame",
    "trailer floor",
    "jack",
    "fender",
    "shadow",
]

IMAGE_EXTENSIONS  = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
COLOR_LOWRES      = (30,  144, 255)
COLOR_HIGHRES     = (50,  205,  50)
ALPHA             = 0.45
DISAPPEAR_THRESH  = 0.2   # retention below this → "disappeared"


# ---------------------------------------------------------------------------
# SAM3 helpers
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


def _prompt_to_slug(prompt: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")


def _create_multi_prompt_datapoint(
    pil_image: Image.Image, prompts: list[str], base_query_id: int
) -> tuple[Datapoint, dict[str, int]]:
    w, h = pil_image.size
    dp = Datapoint(find_queries=[], images=[])
    dp.images = [SAMImage(data=pil_image, objects=[], size=[h, w])]
    query_ids: dict[str, int] = {}
    for i, prompt in enumerate(prompts):
        qid = base_query_id + i
        dp.find_queries.append(
            FindQueryLoaded(
                query_text=prompt,
                image_id=0,
                object_ids_output=[],
                is_exhaustive=True,
                query_processing_order=0,
                inference_metadata=InferenceMetadata(
                    coco_image_id=qid,
                    original_image_id=qid,
                    original_category_id=1,
                    original_size=[h, w],
                    object_id=0,
                    frame_index=0,
                ),
            )
        )
        query_ids[prompt] = qid
    return dp, query_ids


def _create_single_datapoint(
    pil_image: Image.Image, prompt: str, query_id: int
) -> Datapoint:
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


def _run_multi_prompt(model, postprocessor, transform, pil_image, prompts,
                      base_qid, device) -> dict[str, dict]:
    dp, query_ids = _create_multi_prompt_datapoint(pil_image, prompts, base_qid)
    dp = transform(dp)
    batch = collate([dp], dict_key="inference")["inference"]
    batch = copy_data_to_device(batch, torch.device(device), non_blocking=True)
    output = model(batch)
    processed = postprocessor.process_results(output, batch.find_metadatas)
    return {p: processed.get(qid) for p, qid in query_ids.items()}


def _run_single(model, postprocessor, transform, pil_image, prompt,
                query_id, device) -> dict | None:
    dp = _create_single_datapoint(pil_image, prompt, query_id)
    dp = transform(dp)
    batch = collate([dp], dict_key="inference")["inference"]
    batch = copy_data_to_device(batch, torch.device(device), non_blocking=True)
    output = model(batch)
    processed = postprocessor.process_results(output, batch.find_metadatas)
    return processed.get(query_id)


# ---------------------------------------------------------------------------
# Crop merging + minimum size enforcement
# ---------------------------------------------------------------------------

def _compute_merged_crops(
    boxes_xyxy,
    img_w: int,
    img_h: int,
    padding_frac: float,
    min_crop_px: int,
) -> list[tuple[int, int, int, int]]:
    """
    Given N instance bboxes, return a list of merged crop coordinates.

    Steps:
      1. Pad each box by padding_frac of its own dimensions.
      2. Iteratively merge any two padded boxes that overlap (greedy union).
         This groups nearby instances (e.g. adjacent chain links) into a
         single crop so they share context during zoom-in inference.
      3. Expand any crop whose shorter side is < min_crop_px, centred on the
         existing crop centre, to give the model enough context.
      4. Clip to image bounds.
    """
    if len(boxes_xyxy) == 0:
        return []

    # Step 1: compute padded boxes
    padded: list[list[int]] = []
    for box in boxes_xyxy:
        x1, y1, x2, y2 = (float(v) for v in box)
        bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)
        px1 = max(0,     int(x1 - bw * padding_frac))
        py1 = max(0,     int(y1 - bh * padding_frac))
        px2 = min(img_w, int(x2 + bw * padding_frac))
        py2 = min(img_h, int(y2 + bh * padding_frac))
        padded.append([px1, py1, px2, py2])

    # Step 2: merge overlapping padded boxes (greedy, repeat until stable)
    changed = True
    while changed:
        changed = False
        merged: list[list[int]] = []
        used = [False] * len(padded)
        for i in range(len(padded)):
            if used[i]:
                continue
            cx1, cy1, cx2, cy2 = padded[i]
            for j in range(i + 1, len(padded)):
                if used[j]:
                    continue
                jx1, jy1, jx2, jy2 = padded[j]
                # Any overlap → merge
                if cx1 < jx2 and cx2 > jx1 and cy1 < jy2 and cy2 > jy1:
                    cx1, cy1 = min(cx1, jx1), min(cy1, jy1)
                    cx2, cy2 = max(cx2, jx2), max(cy2, jy2)
                    used[j] = True
                    changed = True
            used[i] = True
            merged.append([cx1, cy1, cx2, cy2])
        padded = merged

    # Step 3: enforce minimum crop size by expanding around the centre
    final: list[tuple[int, int, int, int]] = []
    for cx1, cy1, cx2, cy2 in padded:
        cw, ch = cx2 - cx1, cy2 - cy1
        if cw < min_crop_px:
            extra = min_crop_px - cw
            cx1 = max(0,     cx1 - extra // 2)
            cx2 = min(img_w, cx2 + (extra - extra // 2))
        if ch < min_crop_px:
            extra = min_crop_px - ch
            cy1 = max(0,     cy1 - extra // 2)
            cy2 = min(img_h, cy2 + (extra - extra // 2))
        final.append((cx1, cy1, cx2, cy2))

    return final


# ---------------------------------------------------------------------------
# Mask / geometry helpers
# ---------------------------------------------------------------------------

def _to_np(masks_tensor: torch.Tensor) -> np.ndarray:
    if masks_tensor.dim() == 4:
        masks_tensor = masks_tensor.squeeze(1)
    return (masks_tensor.cpu().numpy() > 0).astype(np.uint8) * 255


def _flatten(masks_np: np.ndarray) -> np.ndarray:
    return masks_np.any(axis=0).astype(np.uint8) * 255


def _backproject(crop_mask: np.ndarray, crop_coords: tuple, orig_wh: tuple
                 ) -> np.ndarray:
    W, H = orig_wh
    cx1, cy1, cx2, cy2 = crop_coords
    eh, ew = cy2 - cy1, cx2 - cx1
    if crop_mask.shape != (eh, ew):
        crop_mask = np.array(
            Image.fromarray(crop_mask).resize((ew, eh), Image.NEAREST)
        )
    full = np.zeros((H, W), dtype=np.uint8)
    full[cy1:cy2, cx1:cx2] = crop_mask
    return full


def _bbox_area_frac(boxes_xyxy, img_w: int, img_h: int) -> float:
    if len(boxes_xyxy) == 0:
        return 0.0
    img_area = img_w * img_h
    areas = [
        max(0.0, (float(b[2]) - float(b[0])) * (float(b[3]) - float(b[1]))) / img_area
        for b in boxes_xyxy
    ]
    return float(np.mean(areas))


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ab = (a > 0) & (b > 0)
    un = (a > 0) | (b > 0)
    return float(ab.sum()) / float(un.sum()) if un.sum() > 0 else 1.0


def _coverage(mask: np.ndarray) -> float:
    return 100.0 * (mask > 0).sum() / mask.size


def _retention(lr_cov: float, hr_cov: float) -> float:
    """zoom-in coverage / image-level coverage; capped at 2 to avoid div/0 noise."""
    if lr_cov < 1e-6:
        return 1.0
    return min(hr_cov / lr_cov, 2.0)


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def _overlay(pil_image, mask, color, alpha=ALPHA) -> np.ndarray:
    arr = np.array(pil_image.convert("RGB"), dtype=np.float32)
    m = mask > 0
    for c, col in enumerate(color):
        arr[:, :, c] = np.where(m, arr[:, :, c] * (1 - alpha) + col * alpha,
                                arr[:, :, c])
    return np.clip(arr, 0, 255).astype(np.uint8)


def _save_comparison(pil_image, lowres_mask, highres_mask, path,
                     prompt, n_inst, n_crops, lr_cov, hr_cov,
                     iou, ret, mean_bbox_frac) -> None:
    disappeared = ret < DISAPPEAR_THRESH
    status = "  ⚠ DISAPPEARED" if disappeared else ""
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    fig.suptitle(
        f"'{prompt}'  |  {n_inst} instance(s) → {n_crops} merged crop(s)  |  "
        f"bbox {mean_bbox_frac*100:.1f}% of image{status}",
        fontsize=11, y=1.01
    )
    axes[0].imshow(pil_image)
    axes[0].set_title("Original", fontsize=10)

    axes[1].imshow(_overlay(pil_image, lowres_mask, COLOR_LOWRES))
    axes[1].legend(
        handles=[mpatches.Patch(color=[c/255 for c in COLOR_LOWRES],
                                label=f"image-level  cov={lr_cov:.2f}%")],
        loc="lower left", fontsize=8, framealpha=0.7
    )
    axes[1].set_title("Image-level", fontsize=10)

    axes[2].imshow(_overlay(pil_image, highres_mask, COLOR_HIGHRES))
    axes[2].legend(
        handles=[mpatches.Patch(color=[c/255 for c in COLOR_HIGHRES],
                                label=f"zoom-in  cov={hr_cov:.2f}%  "
                                      f"IoU={iou:.3f}  retain={ret:.2f}")],
        loc="lower left", fontsize=8, framealpha=0.7
    )
    axes[2].set_title("Zoom-in (back-projected)", fontsize=10)

    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _save_chart(per_prompt: dict[str, dict], output_path: str,
                disappear_thresh: float) -> None:
    labels, ious, delta_covs, retentions, disap_rates, bbox_fracs, det_rates = \
        [], [], [], [], [], [], []

    for prompt, s in per_prompt.items():
        if s["n_detected_images"] == 0:
            continue
        labels.append(prompt)
        ious.append(s["mean_iou"])
        delta_covs.append(s["mean_highres_cov"] - s["mean_lowres_cov"])
        retentions.append(s["mean_retention"])
        disap_rates.append(s["disappearance_rate"] * 100)
        bbox_fracs.append(s["mean_bbox_area_frac"] * 100)
        det_rates.append(100.0 * s["n_detected_images"] / s["n_total_images"])

    if not labels:
        return

    order = np.argsort(bbox_fracs)
    labels, ious, delta_covs, retentions = \
        [[v[i] for i in order] for v in [labels, ious, delta_covs, retentions]]
    disap_rates, bbox_fracs, det_rates = \
        [[v[i] for i in order] for v in [disap_rates, bbox_fracs, det_rates]]

    x = np.arange(len(labels))
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    fig.suptitle(
        "Zoom-in impact by prompt  (sorted by mean instance bbox size, smallest→largest)",
        fontsize=13, y=1.005
    )

    # Panel 1: IoU
    b1 = axes[0].bar(x, ious, color="steelblue", alpha=0.8, edgecolor="k", lw=0.5)
    axes[0].axhline(1.0, color="gray", lw=0.8, ls="--")
    axes[0].set_ylabel("Mean IoU\n(lower = zoom differs more)", fontsize=9)
    axes[0].set_ylim(0, 1.1)
    axes[0].bar_label(b1, fmt="%.2f", fontsize=7, padding=2)

    # Panel 2: mean retention (zoom coverage / image-level coverage)
    colors2 = ["seagreen" if r >= 0.8 else ("orange" if r >= 0.2 else "salmon")
               for r in retentions]
    b2 = axes[1].bar(x, retentions, color=colors2, alpha=0.85, edgecolor="k", lw=0.5)
    axes[1].axhline(1.0, color="gray", lw=0.8, ls="--", label="perfect retention")
    axes[1].axhline(disappear_thresh, color="red", lw=1, ls=":", label=f"disappear threshold ({disappear_thresh:.0%})")
    axes[1].set_ylabel("Mean retention\n(zoom cov / image-level cov)", fontsize=9)
    axes[1].set_ylim(0, max(max(retentions) * 1.15, 1.2))
    axes[1].legend(fontsize=7, loc="upper right")
    axes[1].bar_label(b2, fmt="%.2f", fontsize=7, padding=2)

    # Panel 3: disappearance rate
    b3 = axes[2].bar(x, disap_rates, color="crimson", alpha=0.75, edgecolor="k", lw=0.5)
    axes[2].set_ylabel(f"Disappearance rate %\n(retention < {disappear_thresh:.0%})", fontsize=9)
    axes[2].set_ylim(0, 105)
    axes[2].bar_label(b3, fmt="%.0f%%", fontsize=7, padding=2)

    # Panel 4: bbox size + detection rate
    ax4b = axes[3].twinx()
    b4 = axes[3].bar(x, bbox_fracs, color="goldenrod", alpha=0.7,
                     edgecolor="k", lw=0.5, label="mean bbox area %")
    ax4b.plot(x, det_rates, "o-", color="navy", ms=5, lw=1.5, label="detection rate %")
    axes[3].set_ylabel("Mean bbox area\n(% of image)", fontsize=9)
    ax4b.set_ylabel("Detection rate %", fontsize=9, color="navy")
    ax4b.tick_params(axis="y", colors="navy")
    ax4b.set_ylim(0, 115)
    axes[3].bar_label(b4, fmt="%.2f%%", fontsize=6, padding=2)
    h1, l1 = axes[3].get_legend_handles_labels()
    h2, l2 = ax4b.get_legend_handles_labels()
    axes[3].legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)

    axes[3].set_xticks(x)
    axes[3].set_xticklabels(labels, rotation=35, ha="right", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved: {output_path}")


# ---------------------------------------------------------------------------
# Per-image evaluation
# ---------------------------------------------------------------------------

def evaluate_image(
    model, postprocessor, transform,
    pil_image: Image.Image,
    prompts: list[str],
    padding_frac: float,
    min_crop_px: int,
    disappear_thresh: float,
    device: str,
    img_idx: int,
    output_dir: str,
    stem: str,
    save_comparisons: bool,
) -> dict[str, dict]:
    W, H = pil_image.size

    # ── Pass 1: all prompts, one forward pass ─────────────────────────────
    base_qid = img_idx * (len(prompts) + 2000)
    prompt_results = _run_multi_prompt(
        model, postprocessor, transform, pil_image, prompts, base_qid, device
    )

    zoom_qid = base_qid + len(prompts) + 1
    results_by_prompt: dict[str, dict] = {}

    for prompt, result in prompt_results.items():
        slug = _prompt_to_slug(prompt)

        has_det = (
            result is not None
            and "masks" in result
            and len(result["masks"]) > 0
        )
        if not has_det:
            results_by_prompt[prompt] = {"no_detection": True}
            continue

        masks_np = _to_np(result["masks"])    # [N, H, W]
        boxes    = result["boxes"].cpu()      # [N, 4] XYXY
        n_inst   = len(masks_np)
        lowres_mask     = _flatten(masks_np)
        mean_bbox_frac  = _bbox_area_frac(boxes, W, H)

        # ── Pass 2: merged crops ──────────────────────────────────────────
        crop_coords_list = _compute_merged_crops(
            boxes, W, H, padding_frac, min_crop_px
        )
        n_crops = len(crop_coords_list)

        highres_full = np.zeros((H, W), dtype=np.uint8)
        for crop_coords in crop_coords_list:
            cx1, cy1, cx2, cy2 = crop_coords
            pil_crop = pil_image.crop((cx1, cy1, cx2, cy2))

            result_zoom = _run_single(
                model, postprocessor, transform, pil_crop,
                prompt, zoom_qid, device
            )
            zoom_qid += 1

            if (result_zoom is not None
                    and "masks" in result_zoom
                    and len(result_zoom["masks"]) > 0):
                zoom_flat = _flatten(_to_np(result_zoom["masks"]))
                highres_full = np.maximum(
                    highres_full, _backproject(zoom_flat, crop_coords, (W, H))
                )

        lr_cov  = _coverage(lowres_mask)
        hr_cov  = _coverage(highres_full)
        iou_val = _iou(lowres_mask, highres_full)
        ret_val = _retention(lr_cov, hr_cov)
        disappeared = ret_val < disappear_thresh

        if save_comparisons:
            comp_path = os.path.join(output_dir, f"{stem}__{slug}_comp.jpg")
            _save_comparison(
                pil_image, lowres_mask, highres_full, comp_path,
                prompt=prompt, n_inst=n_inst, n_crops=n_crops,
                lr_cov=lr_cov, hr_cov=hr_cov,
                iou=iou_val, ret=ret_val,
                mean_bbox_frac=mean_bbox_frac,
            )

        results_by_prompt[prompt] = {
            "no_detection":     False,
            "n_instances":      n_inst,
            "n_crops":          n_crops,
            "mean_bbox_area_frac": mean_bbox_frac,
            "lowres_cov":       lr_cov,
            "highres_cov":      hr_cov,
            "iou":              iou_val,
            "retention":        ret_val,
            "disappeared":      disappeared,
        }

    return results_by_prompt


# ---------------------------------------------------------------------------
# Aggregation + summary
# ---------------------------------------------------------------------------

def _aggregate(all_results: list[dict[str, dict]], prompts: list[str],
               n_total: int) -> dict[str, dict]:
    per_prompt: dict[str, dict] = {}
    for prompt in prompts:
        rows = [r[prompt] for r in all_results
                if prompt in r and not r[prompt].get("no_detection")]
        if rows:
            per_prompt[prompt] = {
                "n_total_images":      n_total,
                "n_detected_images":   len(rows),
                "mean_iou":            float(np.mean([r["iou"]               for r in rows])),
                "median_iou":          float(np.median([r["iou"]             for r in rows])),
                "mean_lowres_cov":     float(np.mean([r["lowres_cov"]        for r in rows])),
                "mean_highres_cov":    float(np.mean([r["highres_cov"]       for r in rows])),
                "mean_instances":      float(np.mean([r["n_instances"]       for r in rows])),
                "mean_crops":          float(np.mean([r["n_crops"]           for r in rows])),
                "mean_bbox_area_frac": float(np.mean([r["mean_bbox_area_frac"] for r in rows])),
                "mean_retention":      float(np.mean([r["retention"]         for r in rows])),
                "disappearance_rate":  float(np.mean([r["disappeared"]       for r in rows])),
            }
        else:
            per_prompt[prompt] = {
                "n_total_images": n_total, "n_detected_images": 0,
                "mean_iou": float("nan"), "median_iou": float("nan"),
                "mean_lowres_cov": float("nan"), "mean_highres_cov": float("nan"),
                "mean_instances": float("nan"), "mean_crops": float("nan"),
                "mean_bbox_area_frac": float("nan"),
                "mean_retention": float("nan"), "disappearance_rate": float("nan"),
            }
    return per_prompt


def _write_summary(per_prompt: dict[str, dict], output_dir: str,
                   prompts: list[str], padding_frac: float,
                   min_crop_px: int, threshold: float,
                   disappear_thresh: float, n_images: int) -> None:
    def sort_key(p):
        s = per_prompt[p]
        if s["n_detected_images"] == 0:
            return 1e9
        return s["mean_bbox_area_frac"]

    lines = [
        "Zoom-in analysis — all prompts",
        "=" * 78,
        f"Padding              : {padding_frac:.0%}",
        f"Min crop size        : {min_crop_px}px",
        f"Detection threshold  : {threshold}",
        f"Disappear threshold  : retention < {disappear_thresh:.0%}",
        f"Images               : {n_images}",
        "",
        "Sorted by mean instance bbox area (smallest → largest)",
        "",
        f"  {'Prompt':<18} {'det%':>5}  {'inst':>5}  {'crops':>5}  "
        f"{'bbox%':>6}  {'LR%':>7}  {'HR%':>7}  "
        f"{'IoU':>6}  {'retain':>7}  {'disap%':>7}",
        f"  {'-'*18} {'-'*5}  {'-'*5}  {'-'*5}  {'-'*6}  {'-'*7}  {'-'*7}  "
        f"{'-'*6}  {'-'*7}  {'-'*7}",
    ]

    for prompt in sorted(prompts, key=sort_key):
        s = per_prompt[prompt]
        if s["n_detected_images"] == 0:
            lines.append(f"  {prompt:<18} {'0':>5}%  {'—':>5}  {'—':>5}  "
                         f"{'—':>6}  {'—':>7}  {'—':>7}  {'—':>6}  {'—':>7}  {'—':>7}")
            continue
        det_pct = 100.0 * s["n_detected_images"] / s["n_total_images"]
        flag = " ⚠" if s["disappearance_rate"] > 0.3 else ""
        lines.append(
            f"  {prompt:<18} {det_pct:>5.0f}%  "
            f"{s['mean_instances']:>5.1f}  "
            f"{s['mean_crops']:>5.1f}  "
            f"{s['mean_bbox_area_frac']*100:>6.2f}%  "
            f"{s['mean_lowres_cov']:>7.3f}%  "
            f"{s['mean_highres_cov']:>7.3f}%  "
            f"{s['mean_iou']:>6.3f}  "
            f"{s['mean_retention']:>7.2f}  "
            f"{s['disappearance_rate']*100:>7.1f}%{flag}"
        )

    lines += [
        "",
        "── Column guide ────────────────────────────────────────────────────────────",
        "  det%    : % of images where ≥1 instance was detected at image level",
        "  inst    : mean instances per detected image (before merging)",
        "  crops   : mean merged crops per detected image (after merging)",
        "  bbox%   : mean instance bbox as % of image area (object size proxy)",
        "  LR%     : mean image-level mask coverage %",
        "  HR%     : mean zoom-in mask coverage %",
        "  IoU     : mean IoU(image-level, zoom-in)",
        "            → lower = zoom-in finds more different / granular structure",
        "  retain  : mean (HR% / LR%) — how much of the image-level mask survives",
        "            → < 1  = zoom-in is more conservative",
        "            → >> 1 = zoom-in expands the mask",
        "            → < 0.2 counted as 'disappeared'",
        "  disap%  : % of detected images where retention < threshold  ⚠ > 30%",
        "            → high value = zoom-in is harmful for this prompt",
        "",
        "── Interpretation ──────────────────────────────────────────────────────────",
        textwrap.fill(
            "Small objects (low bbox%) with low IoU and good retention are the "
            "prime candidates for zoom-in. High disappearance rate means the "
            "model loses the object when deprived of its surrounding context — "
            "for those prompts, zoom-in should use larger padding or be skipped. "
            "inst > crops confirms that nearby instances are being merged.",
            width=76, initial_indent="  ", subsequent_indent="  "
        ),
    ]

    text = "\n".join(lines) + "\n"
    path = os.path.join(output_dir, "summary.txt")
    with open(path, "w") as f:
        f.write(text)
    print(f"\n{text}")
    print(f"Summary: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_zoom_all_prompts(
    input_dir: str,
    output_dir: str,
    prompts: list[str] | None = None,
    padding_frac: float = 0.4,
    min_crop_px: int = 300,
    disappear_thresh: float = DISAPPEAR_THRESH,
    threshold: float = 0.3,
    save_comparisons: bool = False,
    max_images: int | None = None,
    device: str = "cuda",
) -> None:
    if prompts is None:
        prompts = DEFAULT_PROMPTS

    image_files = sorted(
        p for p in Path(input_dir).iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if max_images:
        image_files = image_files[:max_images]
    if not image_files:
        print(f"No images found in {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"Found {len(image_files)} images  ({len(prompts)} prompts)")
    print(f"Padding              : {padding_frac:.0%}")
    print(f"Min crop size        : {min_crop_px}px")
    print(f"Disappear threshold  : {disappear_thresh:.0%}")
    print(f"Save comparisons     : {save_comparisons}")
    print(f"Device               : {device}")
    print(f"Output               : {output_dir}")
    print("\nLoading model...")

    model         = build_sam3_image_model(device=device, eval_mode=True, load_from_HF=True)
    transform     = _build_transform()
    postprocessor = _make_postprocessor(threshold)
    all_results: list[dict[str, dict]] = []

    autocast_device = "cuda" if device.startswith("cuda") else "cpu"
    with torch.inference_mode(), torch.autocast(autocast_device, dtype=torch.bfloat16):
        for img_idx, img_path in enumerate(tqdm(image_files, desc="Images")):
            pil_image = Image.open(img_path).convert("RGB")
            results = evaluate_image(
                model, postprocessor, transform, pil_image,
                prompts=prompts,
                padding_frac=padding_frac,
                min_crop_px=min_crop_px,
                disappear_thresh=disappear_thresh,
                device=device,
                img_idx=img_idx,
                output_dir=output_dir,
                stem=img_path.stem,
                save_comparisons=save_comparisons,
            )
            all_results.append(results)

    per_prompt = _aggregate(all_results, prompts, n_total=len(image_files))
    _write_summary(per_prompt, output_dir, prompts, padding_frac, min_crop_px,
                   threshold, disappear_thresh, n_images=len(image_files))
    _save_chart(per_prompt, os.path.join(output_dir, "zoom_impact_chart.jpg"),
                disappear_thresh)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zoom-in segmentation analysis across all prompts, with bbox merging."
    )
    parser.add_argument("--input-dir",  required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompts", default=None,
                        help=f"Comma-separated. Default: all {len(DEFAULT_PROMPTS)} prompts")
    parser.add_argument("--padding",   type=float, default=0.4,
                        help="Padding fraction around each bbox before merging (default 0.4)")
    parser.add_argument("--min-crop-px", type=int, default=300,
                        help="Minimum crop side in pixels after merging (default 300)")
    parser.add_argument("--disappearance-threshold", type=float, default=DISAPPEAR_THRESH,
                        help=f"Retention below this = disappeared (default {DISAPPEAR_THRESH})")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Detection confidence threshold (default 0.3)")
    parser.add_argument("--save-comparisons", action="store_true",
                        help="Save 3-panel comparison JPEGs per detected image×prompt")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_zoom_all_prompts(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        prompts=[p.strip() for p in args.prompts.split(",") if p.strip()]
               if args.prompts else None,
        padding_frac=args.padding,
        min_crop_px=args.min_crop_px,
        disappear_thresh=args.disappearance_threshold,
        threshold=args.threshold,
        save_comparisons=args.save_comparisons,
        max_images=args.max_images,
        device=args.device,
    )


if __name__ == "__main__":
    main()
