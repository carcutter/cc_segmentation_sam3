#!/usr/bin/env python3
"""
Trailer segmentation PSD export tool.

Runs two inference passes per batch of images and composes a multi-layer
Photoshop PSD for each image:

  Pass 1  – finetuned SAM3 model (decoder-only checkpoint) with a single
             "trailer" prompt → high-quality silhouette mask.
  Pass 2  – base SAM3 model with 15 detail prompts in one forward pass per
             image (chain, cable, wheel, …) → per-part supplemental masks.

Layer stack in the saved PSD (top → bottom):
  ┌───────────────────────────────────┐
  │  chain                            │  ← detail layers from base model
  │  safety_chain                     │
  │  cable                            │
  │  …                                │
  │  shadow                           │  ← directly above trailer
  ├───────────────────────────────────┤
  │  trailer [finetuned]              │  ← finetuned trailer silhouette
  ├───────────────────────────────────┤
  │  Input Image  (background)        │  ← original RGB photo
  └───────────────────────────────────┘

Each mask layer is stored as a white RGBA fill with the binary mask as the
alpha channel, so it can be loaded directly as a selection in Photoshop
(Ctrl+click / Cmd+click the layer thumbnail).

Environment:
    conda activate cc_sam3

Usage:
    python carcutter/inference/run_psd_export.py \\
        --input-dir /path/to/images \\
        --output-dir /path/to/psds \\
        --finetuned-checkpoint /path/to/checkpoint_10.pt \\
        [--trailer-prompt "trailer"] \\
        [--prompts "chain,cable,safety chain,..."] \\
        [--trailer-threshold 0.5] \\
        [--base-threshold 0.3] \\
        [--device cuda]
"""

import argparse
import gc
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from psd_tools import PSDImage
from psd_tools.constants import BlendMode

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

DEFAULT_CHECKPOINT = (
    "/home/rutger/work/cc_segmentation_sam3/experiments"
    "/trailer_foreground_crop_20260322/checkpoints/checkpoint_10.pt"
)
DEFAULT_TRAILER_PROMPT = "trailer"
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
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
MASK_OPACITY = 204  # 80 % of 255


# ---------------------------------------------------------------------------
# Shared inference helpers
# ---------------------------------------------------------------------------


def _build_transform() -> ComposeAPI:
    return ComposeAPI(
        transforms=[
            RandomResizeAPI(sizes=1008, max_size=1008, square=True, consistent_transform=False),
            ToTensorAPI(),
            NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def _flatten_masks(masks_tensor: torch.Tensor) -> np.ndarray | None:
    """OR all instance masks into a single binary mask. Returns H×W uint8 (0/255)."""
    if masks_tensor is None or len(masks_tensor) == 0:
        return None
    if masks_tensor.dim() == 4:
        masks_tensor = masks_tensor.squeeze(1)
    flat = masks_tensor.any(dim=0)
    return flat.cpu().numpy().astype(np.uint8) * 255


# ---------------------------------------------------------------------------
# Zoom-in helpers (smart merge)
# ---------------------------------------------------------------------------


def _compute_merged_crops(
    boxes_xyxy,
    img_w: int,
    img_h: int,
    padding_frac: float,
    min_crop_px: int,
) -> list[tuple[int, int, int, int]]:
    """Pad, merge overlapping, and enforce minimum size for instance bboxes."""
    if len(boxes_xyxy) == 0:
        return []

    padded: list[list[int]] = []
    for box in boxes_xyxy:
        x1, y1, x2, y2 = (float(v) for v in box)
        bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)
        padded.append([
            max(0,     int(x1 - bw * padding_frac)),
            max(0,     int(y1 - bh * padding_frac)),
            min(img_w, int(x2 + bw * padding_frac)),
            min(img_h, int(y2 + bh * padding_frac)),
        ])

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
                if cx1 < jx2 and cx2 > jx1 and cy1 < jy2 and cy2 > jy1:
                    cx1, cy1 = min(cx1, jx1), min(cy1, jy1)
                    cx2, cy2 = max(cx2, jx2), max(cy2, jy2)
                    used[j] = True
                    changed = True
            used[i] = True
            merged.append([cx1, cy1, cx2, cy2])
        padded = merged

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


def _smart_merge_zoom(
    model,
    postprocessor: "PostProcessImage",
    transform,
    pil_image: Image.Image,
    prompt: str,
    lr_mask: np.ndarray,
    boxes_tensor: torch.Tensor,
    padding_frac: float,
    min_crop_px: int,
    base_qid: int,
    device: str,
) -> np.ndarray:
    """
    Per-crop zoom-in with smart fallback.

    For each merged crop:
      - Run zoom-in inference on the crop.
      - If zoom detects something that overlaps the image-level mask → use zoom.
      - Otherwise (no detection, or no overlap) → keep image-level mask.

    Returns a refined H×W uint8 (0/255) mask.
    """
    h, w = lr_mask.shape

    if boxes_tensor is None or len(boxes_tensor) == 0:
        return lr_mask

    boxes_np = boxes_tensor.cpu().numpy()
    crops = _compute_merged_crops(boxes_np, w, h, padding_frac, min_crop_px)

    result = lr_mask.copy()

    for crop_idx, (cx1, cy1, cx2, cy2) in enumerate(crops):
        crop_pil = pil_image.crop((cx1, cy1, cx2, cy2))
        crop_h, crop_w = cy2 - cy1, cx2 - cx1

        qid = base_qid + crop_idx
        dp = _create_datapoint(crop_pil, prompt, qid)
        dp = transform(dp)
        batch = collate([dp], dict_key="inference")["inference"]
        batch = copy_data_to_device(batch, torch.device(device), non_blocking=True)
        output = model(batch)
        proc = postprocessor.process_results(output, batch.find_metadatas)
        zoom_result = proc.get(qid)

        has_zoom = (
            zoom_result is not None
            and "masks" in zoom_result
            and len(zoom_result["masks"]) > 0
        )
        if not has_zoom:
            continue  # no detection → keep image-level in this region

        zoom_flat = _flatten_masks(zoom_result["masks"])  # crop_h × crop_w
        if zoom_flat is None:
            continue

        # Resize to crop region size if needed (postprocessor uses crop's original_size)
        if zoom_flat.shape != (crop_h, crop_w):
            zoom_flat = np.array(
                Image.fromarray(zoom_flat).resize((crop_w, crop_h), Image.NEAREST)
            )

        # Check overlap with image-level mask in this crop region
        lr_region = lr_mask[cy1:cy2, cx1:cx2]
        has_overlap = bool(((zoom_flat > 0) & (lr_region > 0)).any())

        if has_overlap:
            # Zoom confirmed by overlap → replace this crop region with zoom result
            result[cy1:cy2, cx1:cx2] = zoom_flat

    return result


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


def _create_multi_prompt_datapoint(
    pil_image: Image.Image,
    prompts: list[str],
    base_query_id: int,
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


# ---------------------------------------------------------------------------
# PSD composition helpers
# ---------------------------------------------------------------------------


def _prompt_to_layer_name(prompt: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")


def _mask_to_rgba(mask: np.ndarray) -> Image.Image:
    """Convert H×W uint8 (0/255) binary mask to RGBA.

    White (255,255,255) fill with alpha = mask, so the layer acts as a pure
    white overlay that is only visible where the segmentation is active.
    Loading the layer thumbnail as a Photoshop selection works correctly.
    """
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = 255  # white fill
    rgba[:, :, 3] = mask  # alpha = segmentation
    return Image.fromarray(rgba, mode="RGBA")


def compose_psd(
    pil_image: Image.Image,
    trailer_mask: np.ndarray,
    prompt_masks: dict[str, np.ndarray],
    output_path: str,
) -> dict:
    """Build and save a multi-layer PSD.

    Layer order (Photoshop panel, top → bottom):
      detail layers (one per prompt, in prompt list order)
      trailer [finetuned]
      Input Image

    Returns a dict with metadata for post-save verification.
    """
    w, h = pil_image.size
    psd = PSDImage.new("RGB", (w, h))

    # psd-tools appends layers bottom→top (first appended = bottom of panel).
    # Desired Photoshop order top→bottom:
    #   chain | … | fender | shadow | trailer [finetuned] | Input Image
    # So we append: Input Image first (bottom), then trailer, then detail layers.

    # Input Image at the bottom (appended first).
    psd.create_pixel_layer(
        pil_image.convert("RGB"),
        name="Input Image",
        blend_mode=BlendMode.NORMAL,
        opacity=255,
    )

    # Trailer silhouette from finetuned model — above the background.
    psd.create_pixel_layer(
        _mask_to_rgba(trailer_mask),
        name="trailer [finetuned]",
        blend_mode=BlendMode.NORMAL,
        opacity=MASK_OPACITY,
    )

    # Detail masks from base model — appended on top.
    # "shadow" is the last key in prompt_masks (tail of DEFAULT_PROMPTS),
    # so it lands directly above trailer [finetuned].
    for prompt, mask in prompt_masks.items():
        layer_name = _prompt_to_layer_name(prompt)
        psd.create_pixel_layer(
            _mask_to_rgba(mask),
            name=layer_name,
            blend_mode=BlendMode.NORMAL,
            opacity=MASK_OPACITY,
        )

    expected_layer_count = 1 + 1 + len(prompt_masks)  # background + trailer + prompts
    psd.save(output_path)
    return {"width": w, "height": h, "expected_layers": expected_layer_count}


def verify_psd(output_path: str, expected: dict) -> list[str]:
    """Reopen the saved PSD and sanity-check it. Returns a list of error strings."""
    errors = []
    file_size = os.path.getsize(output_path)
    if file_size == 0:
        errors.append("file is empty (0 bytes)")
        return errors  # no point opening a zero-byte file

    try:
        psd = PSDImage.open(output_path)
    except Exception as exc:
        errors.append(f"cannot reopen PSD: {exc}")
        return errors

    if psd.width != expected["width"] or psd.height != expected["height"]:
        errors.append(
            f"canvas size mismatch: got {psd.width}×{psd.height}, "
            f"expected {expected['width']}×{expected['height']}"
        )

    actual_layers = len(list(psd))
    if actual_layers != expected["expected_layers"]:
        errors.append(
            f"layer count mismatch: got {actual_layers}, "
            f"expected {expected['expected_layers']}"
        )

    return errors


# ---------------------------------------------------------------------------
# Inference passes
# ---------------------------------------------------------------------------


def _pass1_finetuned(
    image_files: list[Path],
    finetuned_checkpoint: str,
    prompt: str,
    threshold: float,
    device: str,
    tmp_dir: str,
) -> None:
    """Run finetuned model on all images; save trailer masks as PNGs to tmp_dir."""
    print("\n── Pass 1: finetuned model ──────────────────────────────────────")
    print(f"Prompt     : '{prompt}'")
    print(f"Checkpoint : {finetuned_checkpoint}")
    print(f"Threshold  : {threshold}")

    model = build_sam3_image_model(device=device, eval_mode=True, load_from_HF=True)
    ckpt = torch.load(finetuned_checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  checkpoint missing keys : {len(missing)}")
    if unexpected:
        print(f"  checkpoint unexpected keys : {len(unexpected)}")

    transform = _build_transform()
    postprocessor = _make_postprocessor(threshold)
    autocast_device = "cuda" if device.startswith("cuda") else "cpu"

    with torch.inference_mode(), torch.autocast(autocast_device, dtype=torch.bfloat16):
        for query_id, img_path in enumerate(tqdm(image_files, desc="Finetuned"), start=1):
            pil_image = Image.open(img_path).convert("RGB")
            h, w = pil_image.size[1], pil_image.size[0]

            dp = _create_datapoint(pil_image, prompt, query_id)
            dp = transform(dp)
            batch = collate([dp], dict_key="inference")["inference"]
            batch = copy_data_to_device(batch, torch.device(device), non_blocking=True)

            output = model(batch)
            processed = postprocessor.process_results(output, batch.find_metadatas)
            result = processed.get(query_id)

            has_masks = result is not None and "masks" in result and len(result["masks"]) > 0
            flat = _flatten_masks(result["masks"]) if has_masks else np.zeros((h, w), dtype=np.uint8)

            Image.fromarray(flat).save(os.path.join(tmp_dir, img_path.stem + ".png"))

    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    gc.collect()


def _pass2_base_and_compose(
    image_files: list[Path],
    prompts: list[str],
    threshold: float,
    device: str,
    trailer_mask_dir: str,
    output_dir: str,
    padding_frac: float = 0.4,
    min_crop_px: int = 300,
) -> None:
    """Run base model (multi-prompt) + zoom-in smart merge, then compose PSD."""
    print("\n── Pass 2: base model + zoom-in smart merge + PSD composition ───")
    print(f"Prompts      : {prompts}")
    print(f"Threshold    : {threshold}")
    print(f"Zoom padding : {padding_frac}   min crop: {min_crop_px}px")

    model = build_sam3_image_model(
        device=device,
        eval_mode=True,
        checkpoint_path=None,
        load_from_HF=True,
    )

    transform = _build_transform()
    postprocessor = _make_postprocessor(threshold)
    autocast_device = "cuda" if device.startswith("cuda") else "cpu"

    failed: list[str] = []
    ok_sizes: list[float] = []

    with torch.inference_mode(), torch.autocast(autocast_device, dtype=torch.bfloat16):
        for img_idx, img_path in enumerate(tqdm(image_files, desc="Base+PSD")):
            pil_image = Image.open(img_path).convert("RGB")
            h, w = pil_image.size[1], pil_image.size[0]

            # ── image-level multi-prompt pass ───────────────────────────────
            base_query_id = img_idx * len(prompts) * 200 + 1
            dp, query_ids = _create_multi_prompt_datapoint(pil_image, prompts, base_query_id)
            dp = transform(dp)
            batch = collate([dp], dict_key="inference")["inference"]
            batch = copy_data_to_device(batch, torch.device(device), non_blocking=True)

            output = model(batch)
            processed = postprocessor.process_results(output, batch.find_metadatas)

            # ── per-prompt smart-merge zoom ─────────────────────────────────
            prompt_masks: dict[str, np.ndarray] = {}
            for p_idx, (prompt, qid) in enumerate(query_ids.items()):
                result = processed.get(qid)
                has_masks = result is not None and "masks" in result and len(result["masks"]) > 0

                if not has_masks:
                    prompt_masks[prompt] = np.zeros((h, w), dtype=np.uint8)
                    continue

                lr_mask = _flatten_masks(result["masks"])
                boxes = result.get("boxes")  # N×4 xyxy in original image coords

                if boxes is not None and len(boxes) > 0:
                    # Unique QID block for zoom crops: offset per image+prompt
                    zoom_base_qid = (
                        img_idx * len(prompts) * 200
                        + p_idx * 100
                        + len(prompts) + 1
                    )
                    lr_mask = _smart_merge_zoom(
                        model=model,
                        postprocessor=postprocessor,
                        transform=transform,
                        pil_image=pil_image,
                        prompt=prompt,
                        lr_mask=lr_mask,
                        boxes_tensor=boxes,
                        padding_frac=padding_frac,
                        min_crop_px=min_crop_px,
                        base_qid=zoom_base_qid,
                        device=device,
                    )

                prompt_masks[prompt] = lr_mask

            # Load trailer mask saved in pass 1.
            trailer_mask_path = os.path.join(trailer_mask_dir, img_path.stem + ".png")
            trailer_mask = np.array(Image.open(trailer_mask_path).convert("L"))

            output_path = os.path.join(output_dir, img_path.stem + ".psd")
            meta = compose_psd(pil_image, trailer_mask, prompt_masks, output_path)

            errors = verify_psd(output_path, meta)
            if errors:
                tqdm.write(f"  WARNING {img_path.name}: {'; '.join(errors)}")
                failed.append(img_path.name)
            else:
                file_kb = os.path.getsize(output_path) / 1024
                ok_sizes.append(file_kb)


    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    gc.collect()

    # ── verification summary ────────────────────────────────────────────────
    n_ok = len(ok_sizes)
    n_fail = len(failed)
    print(f"\n{'=' * 60}")
    print(f"PSD export complete")
    print(f"  OK      : {n_ok}/{len(image_files)}")
    if ok_sizes:
        print(
            f"  Size    : {min(ok_sizes):.0f} KB – {max(ok_sizes):.0f} KB "
            f"(avg {sum(ok_sizes)/len(ok_sizes):.0f} KB)"
        )
        # Report expected layer count (same for all images)
        n_layers = 1 + 1 + len(prompts)
        print(f"  Layers  : {n_layers}  (Input Image + trailer [finetuned] + {len(prompts)} detail layers)")
    if failed:
        print(f"  FAILED  : {n_fail}")
        for name in failed:
            print(f"    • {name}")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_psd_export(
    input_dir: str,
    output_dir: str,
    finetuned_checkpoint: str,
    trailer_prompt: str = DEFAULT_TRAILER_PROMPT,
    prompts: list[str] | None = None,
    trailer_threshold: float = 0.5,
    base_threshold: float = 0.3,
    device: str = "cuda",
    padding_frac: float = 0.4,
    min_crop_px: int = 300,
) -> None:
    if prompts is None:
        prompts = DEFAULT_PROMPTS

    image_files = sorted(
        p for p in Path(input_dir).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_files:
        print(f"No images found in {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    print(f"Found {len(image_files)} images")
    print(f"Output    : {output_dir}")
    print(f"Device    : {device}")

    with tempfile.TemporaryDirectory(prefix="sam3_trailer_masks_") as tmp_dir:
        _pass1_finetuned(
            image_files=image_files,
            finetuned_checkpoint=finetuned_checkpoint,
            prompt=trailer_prompt,
            threshold=trailer_threshold,
            device=device,
            tmp_dir=tmp_dir,
        )
        _pass2_base_and_compose(
            image_files=image_files,
            prompts=prompts,
            threshold=base_threshold,
            device=device,
            trailer_mask_dir=tmp_dir,
            output_dir=output_dir,
            padding_frac=padding_frac,
            min_crop_px=min_crop_px,
        )

    print(f"\nDone. PSDs saved to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export multi-layer Photoshop PSDs combining a finetuned trailer "
            "segmentation mask with base-model detail masks (chain, cable, …)."
        )
    )
    parser.add_argument("--input-dir", required=True, help="Folder of input images.")
    parser.add_argument(
        "--output-dir", required=True, help="Folder where .psd files will be written."
    )
    parser.add_argument(
        "--finetuned-checkpoint",
        default=DEFAULT_CHECKPOINT,
        help=f"Path to finetuned trailer checkpoint .pt (default: {DEFAULT_CHECKPOINT}).",
    )
    parser.add_argument(
        "--trailer-prompt",
        default=DEFAULT_TRAILER_PROMPT,
        help=f"Text prompt for the finetuned model (default: '{DEFAULT_TRAILER_PROMPT}').",
    )
    parser.add_argument(
        "--prompts",
        default=None,
        help=(
            f"Comma-separated prompts for the base model detail layers. "
            f"Default: {','.join(DEFAULT_PROMPTS)}"
        ),
    )
    parser.add_argument(
        "--trailer-threshold",
        type=float,
        default=0.5,
        help="Detection threshold for the finetuned trailer model (default: 0.5).",
    )
    parser.add_argument(
        "--base-threshold",
        type=float,
        default=0.3,
        help="Detection threshold for the base model detail prompts (default: 0.3).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (default: cuda if available, else cpu).",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.4,
        help="Fractional padding around each bbox before zoom-in (default: 0.4).",
    )
    parser.add_argument(
        "--min-crop-px",
        type=int,
        default=300,
        help="Minimum zoom-in crop side length in pixels (default: 300).",
    )
    args = parser.parse_args()

    prompts = (
        [p.strip() for p in args.prompts.split(",") if p.strip()]
        if args.prompts
        else None
    )

    run_psd_export(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        finetuned_checkpoint=args.finetuned_checkpoint,
        trailer_prompt=args.trailer_prompt,
        prompts=prompts,
        trailer_threshold=args.trailer_threshold,
        base_threshold=args.base_threshold,
        device=args.device,
        padding_frac=args.padding,
        min_crop_px=args.min_crop_px,
    )


if __name__ == "__main__":
    main()
