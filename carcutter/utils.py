import argparse
import glob
import io
import os

import cv2
import numpy as np
from PIL import Image

TARGET_SIZE = 1008


def get_args():
    parser = argparse.ArgumentParser(description="SAM3 ONNX Inference")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--text", type=str, help="Text prompt")
    parser.add_argument(
        "--boxes", type=str, help="Box prompts: pos:x,y,w,h;neg:x,y,w,h (xywh format)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="facebook/sam3",
        help="Path to SAM3 model (HuggingFace model ID or local path)",
    )
    parser.add_argument(
        "--model-dir", type=str, default="onnx-models", help="ONNX models directory"
    )
    parser.add_argument("--tokenizer", type=str, help="Path to tokenizer.json")
    parser.add_argument("--output", type=str, default="output.png", help="Output path")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    if not args.text and not args.boxes:
        parser.error("Please specify --text or --boxes")

    if not args.tokenizer:
        try:
            tokenizer_dir = os.path.join(
                os.environ["HF_HUB_CACHE"], "models--facebook--sam3", "snapshots"
            )
            args.tokenizer = glob.glob(
                os.path.join(tokenizer_dir, "**", "tokenizer.json"), recursive=True
            )[0]
        except Exception as e:
            print(f"Error finding tokenizer: at {tokenizer_dir}: {e}")
            parser.error("Please specify --tokenizer")

    return args


def visualize_results(
    image_path: str, results: dict, output_path: str, alpha: float = 0.35
):
    """Visualize detection results with mask overlay and contours"""
    vis = cv2.imread(image_path)
    if vis is None:
        raise ValueError(f"Could not load image: {image_path}")
    vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    colors = [
        (30, 144, 255),  # Dodger Blue
        (255, 144, 30),  # Orange
        (144, 255, 30),  # Green-Yellow
        (255, 30, 144),  # Pink
        (30, 255, 144),  # Spring Green
    ]

    for i in range(results["num_detections"][0]):
        color = colors[i % len(colors)]
        score = (
            results["scores"][0][i]
            if len(results["scores"].shape) > 1
            else results["scores"][i]
        )
        box = (
            results["boxes"][0][i]
            if len(results["boxes"].shape) > 2
            else results["boxes"][i]
        )
        x1, y1, x2, y2 = map(int, box)
        mask_bool = results["masks"][0][i] > 0

        # Apply mask overlay (lighter)
        overlay = vis.copy()
        overlay[mask_bool] = color
        vis = cv2.addWeighted(vis, 1 - alpha, overlay, alpha, 0)

        # Draw mask contours
        mask_uint8 = mask_bool.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis, contours, -1, color, 2)

        # Draw box and score
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            vis, f"{score:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )

    cv2.imwrite(output_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"  ✓ Saved: {output_path}")


def parse_box_prompts(box_str: str) -> tuple[list, list]:
    """Parse box prompts string

    Format: "pos:x,y,w,h;neg:x,y,w,h;..." (xywh format)
    Returns: boxes [[x,y,w,h], ...], labels [1, 0, ...]
    """
    boxes, labels = [], []
    for part in box_str.split(";"):
        part = part.strip()
        if not part:
            continue
        if part.startswith("pos:"):
            label, coords = 1, part[4:]
        elif part.startswith("neg:"):
            label, coords = 0, part[4:]
        else:
            label, coords = 1, part  # default positive
        x, y, w, h = [float(v) for v in coords.split(",")]
        boxes.append([x, y, w, h])
        labels.append(label)
    return boxes, labels


def xywh_to_cxcywh_normalized(boxes: list, img_w: int, img_h: int) -> np.ndarray:
    """Convert xywh (pixel) to cxcywh (normalized)"""
    result = []
    for x, y, w, h in boxes:
        cx = (x + w / 2) / img_w
        cy = (y + h / 2) / img_h
        nw = w / img_w
        nh = h / img_h
        result.append([cx, cy, nw, nh])
    return np.array(result, dtype=np.float32)


def preprocess_image(image_bytes: bytes) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Preprocess image: decode, resize, normalize.

    Args:
        image_bytes: Raw image bytes

    Returns:
        Tuple of (preprocessed_image, original_size)
    """
    # Decode image
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig_h, orig_w = image.size[1], image.size[0]  # (height, width)

    # Resize to target size
    image_resized = image.resize((TARGET_SIZE, TARGET_SIZE), Image.BILINEAR)

    # Convert to numpy and normalize
    image_array = np.array(image_resized, dtype=np.float32)
    # Normalize: [0, 255] -> [-1, 1]
    image_normalized = image_array / 127.5 - 1.0

    # Convert to CHW format
    image_chw = image_normalized.transpose(2, 0, 1)  # HWC -> CHW

    # Add batch dimension
    image_batch = np.expand_dims(image_chw, axis=0)  # [1, 3, 1008, 1008]

    return image_batch, (orig_h, orig_w)


def postprocess(
    decoder_outputs: dict, orig_h: int, orig_w: int, conf_threshold: float
) -> dict:
    pred_masks = decoder_outputs["pred_masks"]  # torch.Tensor on GPU
    pred_boxes = decoder_outputs["pred_boxes"]  # torch.Tensor on GPU
    pred_logits = decoder_outputs["pred_logits"]  # torch.Tensor on GPU
    presence_logits = decoder_outputs["presence_logits"]  # torch.Tensor on GPU

    # Post-process outputs (using PyTorch operations on GPU)
    batch_size = pred_masks.shape[0]
    all_masks = []
    all_boxes = []
    all_scores = []

    for b in range(batch_size):
        batch_masks = pred_masks[b]  # [Q, 288, 288] - torch.Tensor on GPU
        batch_boxes = pred_boxes[b]  # [Q, 4] - torch.Tensor on GPU
        batch_logits = pred_logits[b]  # [Q] - torch.Tensor on GPU
        if len(presence_logits.shape) > 1:
            batch_presence = presence_logits[b, 0]
        else:
            batch_presence = presence_logits[b]

        presence_score = 1 / (1 + np.exp(-batch_presence))
        scores = (1 / (1 + np.exp(-batch_logits))) * presence_score

        # Filter by confidence threshold
        keep = scores > conf_threshold

        if keep.sum() == 0:
            all_masks.append(np.array([], dtype=bool))
            all_boxes.append(np.array([], dtype=np.float32))
            all_scores.append(np.array([], dtype=np.float32))
            continue

        # Move filtered tensors to CPU for resizing
        masks = batch_masks[keep]  # Still on GPU
        boxes = batch_boxes[keep]  # Still on GPU
        scores = scores[keep]  # Still on GPU

        # Resize masks to original size (need CPU for cv2)
        filtered_masks = []
        for idx in range(keep.sum().item()):
            mask_resized = cv2.resize(
                masks[idx], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
            )  # [288, 288]
            filtered_masks.append(mask_resized > 0)

        # Scale boxes from normalized [0,1] to pixel coordinates (on GPU, then CPU)
        boxes[:, [0, 2]] *= orig_w
        boxes[:, [1, 3]] *= orig_h
        boxes = np.clip(boxes, 0, [[orig_w, orig_h, orig_w, orig_h]])

        all_masks.append(filtered_masks)
        all_boxes.append(boxes)
        all_scores.append(scores)

    # Create output tensors
    output_tensors = {
        "masks": np.array(all_masks),
        "boxes": np.array(all_boxes),
        "scores": np.array(all_scores),
        "num_detections": np.array([len(m) for m in all_masks]),
    }
    return output_tensors
