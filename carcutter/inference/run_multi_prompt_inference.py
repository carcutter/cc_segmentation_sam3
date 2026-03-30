#!/usr/bin/env python3
"""
Multi-prompt SAM3 inference script.

Runs the base SAM3 model with N text prompts in a **single forward pass per image**
(all prompts packed into one Datapoint), and saves per-prompt binary masks to
separate subfolders under output_dir.

This is ~N× faster than calling run_inference.py N times because SAM3 supports
multiple find_queries in a single Datapoint — the vision backbone runs once.

Usage:
    python carcutter/inference/run_multi_prompt_inference.py \\
        --input-dir /path/to/images \\
        --output-dir /path/to/multi_prompt_preds \\
        [--prompts "chain,cable,wheel,tire,axle,hitch,coupler,ramp,gate,frame,floor,jack,fender"] \\
        [--threshold 0.3] \\
        [--device cuda]

Output structure:
    output_dir/
      chain/            image0001.png, image0002.png, ...
      safety_chain/     image0001.png, ...
      wheel/            ...
      _manifest.json    {slug: {prompt, num_images, num_with_detections}}
"""

import argparse
import json
import os
import re
from pathlib import Path

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
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def prompt_to_slug(prompt: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")
    return slug


def build_transform():
    return ComposeAPI(
        transforms=[
            RandomResizeAPI(sizes=1008, max_size=1008, square=True, consistent_transform=False),
            ToTensorAPI(),
            NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def flatten_masks(masks_tensor: torch.Tensor) -> np.ndarray | None:
    """OR all predicted instance masks into one binary mask. Returns H×W uint8 array (0/255)."""
    if masks_tensor is None or len(masks_tensor) == 0:
        return None
    if masks_tensor.dim() == 4:
        masks_tensor = masks_tensor.squeeze(1)  # [N, 1, H, W] → [N, H, W]
    flat = masks_tensor.any(dim=0)
    return flat.cpu().numpy().astype(np.uint8) * 255


def create_multi_prompt_datapoint(
    pil_image: Image.Image,
    prompts: list[str],
    base_query_id: int,
) -> tuple[Datapoint, dict[str, int]]:
    """
    Pack N prompts into one Datapoint so SAM3 runs a single forward pass.
    Returns the datapoint and a {prompt: query_id} map for result lookup.
    """
    w, h = pil_image.size
    datapoint = Datapoint(find_queries=[], images=[])
    datapoint.images = [SAMImage(data=pil_image, objects=[], size=[h, w])]

    query_ids: dict[str, int] = {}
    for i, prompt in enumerate(prompts):
        qid = base_query_id + i
        datapoint.find_queries.append(
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

    return datapoint, query_ids


def run_multi_prompt_inference(
    input_dir: str,
    output_dir: str,
    prompts: list[str],
    threshold: float = 0.3,
    device: str = "cuda",
) -> None:
    slugs = {p: prompt_to_slug(p) for p in prompts}

    for slug in slugs.values():
        os.makedirs(os.path.join(output_dir, slug), exist_ok=True)

    image_files = sorted(
        p for p in Path(input_dir).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_files:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_files)} images, {len(prompts)} prompts")
    print(f"Prompts   : {prompts}")
    print(f"Threshold : {threshold}")
    print(f"Device    : {device}")
    print(f"Output    : {output_dir}")

    print("\nLoading model...")
    model = build_sam3_image_model(
        device=device,
        eval_mode=True,
        checkpoint_path=None,
        load_from_HF=True,
    )

    transform = build_transform()
    postprocessor = PostProcessImage(
        max_dets_per_img=-1,
        iou_type="segm",
        use_original_sizes_box=True,
        use_original_sizes_mask=True,
        convert_mask_to_rle=False,
        detection_threshold=threshold,
        to_cpu=False,
    )

    detection_counts = {p: 0 for p in prompts}

    autocast_device = "cuda" if device.startswith("cuda") else "cpu"
    with torch.inference_mode(), torch.autocast(autocast_device, dtype=torch.bfloat16):
        for img_idx, img_path in enumerate(tqdm(image_files, desc="Inferring")):
            pil_image = Image.open(img_path).convert("RGB")
            h, w = pil_image.size[1], pil_image.size[0]

            # query IDs must be globally unique across all images × prompts
            base_query_id = img_idx * len(prompts) + 1

            datapoint, query_ids = create_multi_prompt_datapoint(pil_image, prompts, base_query_id)
            datapoint = transform(datapoint)

            batch = collate([datapoint], dict_key="inference")["inference"]
            batch = copy_data_to_device(batch, torch.device(device), non_blocking=True)

            output = model(batch)
            processed = postprocessor.process_results(output, batch.find_metadatas)

            for prompt, qid in query_ids.items():
                result = processed.get(qid)
                slug = slugs[prompt]
                output_path = os.path.join(output_dir, slug, img_path.stem + ".png")

                has_masks = (
                    result is not None
                    and "masks" in result
                    and len(result["masks"]) > 0
                )
                if has_masks:
                    flat = flatten_masks(result["masks"])
                    detection_counts[prompt] += 1
                else:
                    flat = np.zeros((h, w), dtype=np.uint8)

                Image.fromarray(flat).save(output_path)

    manifest = {
        slugs[p]: {
            "prompt": p,
            "slug": slugs[p],
            "num_images": len(image_files),
            "num_with_detections": detection_counts[p],
        }
        for p in prompts
    }
    manifest_path = os.path.join(output_dir, "_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Done. {len(image_files)} images × {len(prompts)} prompts")
    print(f"\n{'Prompt':<22} {'Detections':>12}  {'%':>5}")
    print(f"{'-' * 42}")
    for p in prompts:
        n = detection_counts[p]
        pct = 100.0 * n / len(image_files) if image_files else 0.0
        print(f"  {p:<20} {n:>12,}  {pct:>5.1f}%")
    print(f"{'=' * 60}")
    print(f"Manifest: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run base SAM3 with N text prompts in one forward pass per image. "
            "Saves per-prompt binary mask PNGs to output_dir/<prompt_slug>/."
        )
    )
    parser.add_argument("--input-dir", required=True, help="Folder of input images.")
    parser.add_argument(
        "--output-dir", required=True,
        help="Root folder; per-prompt masks go into subfolders named by prompt slug.",
    )
    parser.add_argument(
        "--prompts", default=None,
        help=f"Comma-separated prompts. Default: {','.join(DEFAULT_PROMPTS)}",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.3,
        help="Detection confidence threshold (default: 0.3).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (default: cuda if available, else cpu).",
    )
    args = parser.parse_args()

    prompts = (
        [p.strip() for p in args.prompts.split(",") if p.strip()]
        if args.prompts
        else DEFAULT_PROMPTS
    )

    run_multi_prompt_inference(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        prompts=prompts,
        threshold=args.threshold,
        device=args.device,
    )


if __name__ == "__main__":
    main()
