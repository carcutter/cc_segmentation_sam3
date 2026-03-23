#!/usr/bin/env python3
"""
Generic SAM3 inference script.

Runs SAM3 on all images in an input folder using a text prompt and saves
flattened binary segmentation masks to an output folder (one PNG per image).

All predicted instances are OR-ed into a single binary mask per image.
Output masks are the same resolution as the input images.

Images are passed to the model as-is (no task-specific cropping or foreground
masking). The model itself resizes internally to 1008×1008 for its vision
backbone and resizes the predicted masks back to the original resolution.

This script is intentionally separate from any ground-truth comparison step —
the output folder can be passed to a comparison script independently.

Usage:
    python carcutter/inference/run_inference.py \\
        --input-dir /path/to/images \\
        --output-dir /path/to/output_masks \\
        --prompt "mirror"

    # With a fine-tuned checkpoint:
    python carcutter/inference/run_inference.py \\
        --input-dir /path/to/images \\
        --output-dir /path/to/output_masks \\
        --prompt "mirror" \\
        --checkpoint /path/to/checkpoint.pt

    # Lower threshold to get more (less confident) detections:
    python carcutter/inference/run_inference.py \\
        --input-dir /path/to/images \\
        --output-dir /path/to/output_masks \\
        --prompt "car" \\
        --threshold 0.3
"""

import os
import argparse
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


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def build_transform():
    return ComposeAPI(
        transforms=[
            RandomResizeAPI(sizes=1008, max_size=1008, square=True, consistent_transform=False),
            ToTensorAPI(),
            NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def create_datapoint(pil_image: Image.Image, text_prompt: str, query_id: int) -> Datapoint:
    w, h = pil_image.size
    datapoint = Datapoint(find_queries=[], images=[])
    datapoint.images = [SAMImage(data=pil_image, objects=[], size=[h, w])]
    datapoint.find_queries.append(
        FindQueryLoaded(
            query_text=text_prompt,
            image_id=0,
            object_ids_output=[],
            is_exhaustive=True,
            query_processing_order=0,
            inference_metadata=InferenceMetadata(
                coco_image_id=query_id,
                original_image_id=query_id,
                original_category_id=1,
                original_size=[w, h],
                object_id=0,
                frame_index=0,
            ),
        )
    )
    return datapoint


def flatten_masks(masks_tensor: torch.Tensor) -> np.ndarray | None:
    """OR all predicted instance masks into one binary mask. Returns H×W uint8 array (0/255)."""
    if masks_tensor is None or len(masks_tensor) == 0:
        return None
    flat = masks_tensor.any(dim=0)  # [N, H, W] → [H, W]
    return (flat.cpu().numpy().astype(np.uint8)) * 255


def run_inference(
    input_dir: str,
    output_dir: str,
    prompt: str,
    checkpoint: str | None = None,
    threshold: float = 0.5,
    device: str = "cuda",
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted(
        p for p in Path(input_dir).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_files:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_files)} images")
    print(f"Prompt      : '{prompt}'")
    print(f"Checkpoint  : {checkpoint or 'base model (HuggingFace)'}")
    print(f"Threshold   : {threshold}")
    print(f"Device      : {device}")
    print(f"Output      : {output_dir}")

    print("\nLoading model...")
    model = build_sam3_image_model(
        device=device,
        eval_mode=True,
        checkpoint_path=checkpoint,
        load_from_HF=(checkpoint is None),
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

    stats = {"total": 0, "with_masks": 0, "empty": 0}

    autocast_device = "cuda" if device.startswith("cuda") else "cpu"
    with torch.inference_mode(), torch.autocast(autocast_device, dtype=torch.bfloat16):
        for query_id, img_path in enumerate(tqdm(image_files, desc="Inferring"), start=1):
            pil_image = Image.open(img_path).convert("RGB")
            h, w = pil_image.size[1], pil_image.size[0]
            stats["total"] += 1

            datapoint = create_datapoint(pil_image, prompt, query_id)
            datapoint = transform(datapoint)

            batch = collate([datapoint], dict_key="inference")["inference"]
            batch = copy_data_to_device(batch, torch.device(device), non_blocking=True)

            output = model(batch)
            processed = postprocessor.process_results(output, batch.find_metadatas)

            result = processed.get(query_id)
            output_path = os.path.join(output_dir, img_path.stem + ".png")

            has_masks = (
                result is not None
                and "masks" in result
                and len(result["masks"]) > 0
            )
            if has_masks:
                flat = flatten_masks(result["masks"])
            else:
                flat = None

            if flat is not None:
                stats["with_masks"] += 1
            else:
                flat = np.zeros((h, w), dtype=np.uint8)
                stats["empty"] += 1

            Image.fromarray(flat).save(output_path)

    print(f"\n{'=' * 60}")
    print(f"Done. Processed {stats['total']} images.")
    print(f"  With detections : {stats['with_masks']}")
    print(f"  Empty           : {stats['empty']}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run SAM3 inference on a folder of images and save binary segmentation masks. "
            "All predicted instances for the given text prompt are OR-ed into a single "
            "binary PNG per image at the original resolution."
        )
    )
    parser.add_argument(
        "--input-dir", required=True,
        help="Folder containing input images.",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Folder where binary mask PNGs will be saved (one per input image).",
    )
    parser.add_argument(
        "--prompt", required=True,
        help="Text prompt describing what to segment, e.g. 'mirror', 'car', 'wheel'.",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help=(
            "Path to a fine-tuned SAM3 checkpoint (.pt). "
            "If omitted, the base model is downloaded from HuggingFace (facebook/sam3)."
        ),
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Detection confidence threshold (default: 0.5). Lower values yield more detections.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (default: cuda if available, else cpu).",
    )
    args = parser.parse_args()

    run_inference(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        prompt=args.prompt,
        checkpoint=args.checkpoint,
        threshold=args.threshold,
        device=args.device,
    )


if __name__ == "__main__":
    main()
