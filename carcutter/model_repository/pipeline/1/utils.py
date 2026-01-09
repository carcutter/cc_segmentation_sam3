"""
Triton Python Backend for SAM3 Inference Pipeline
Orchestrates vision-encoder, text-encoder, geometry-encoder, and decoder models
"""

import io

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from PIL import Image
from torch.utils.dlpack import from_dlpack, to_dlpack

TARGET_SIZE = 1008
PAD_TOKEN_ID = 49407
MAX_TEXT_LENGTH = 32
DEVICE = "cuda"
# Output mapping for clarity
OUTPUT_NAMES = {
    "vision-encoder": ["fpn_feat_0", "fpn_feat_1", "fpn_feat_2", "fpn_pos_2"],
    "text-encoder": ["text_features", "text_mask"],
    "geometry-encoder": ["geometry_features", "geometry_mask"],
    "decoder": ["pred_masks", "pred_boxes", "pred_logits", "presence_logits"],
}


def torch_to_pb_tensor(name: str, tensor: torch.Tensor) -> "pb_utils.Tensor":
    """
    Zero-copy conversion from Torch GPU Tensor to Triton Tensor via DLPack.
    """
    # Ensure tensor is contiguous before converting to DLPack
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return pb_utils.Tensor.from_dlpack(name, to_dlpack(tensor))


def pb_tensor_to_torch(pb_tensor: "pb_utils.Tensor") -> "torch.Tensor":
    """
    Zero-copy conversion from Triton Tensor to Torch GPU Tensor via DLPack.
    """
    if pb_tensor is None:
        return None
    # from_dlpack captures the memory capsule directly
    return from_dlpack(pb_tensor.to_dlpack())


def call_model(
    model_name: str,
    inputs: dict[str, "torch.Tensor"],
) -> dict[str, "torch.Tensor"]:
    """
    Executes a BLS call keeping data on GPU.
    """
    # 1. Prepare inputs (Torch -> DLPack -> PB Tensor)
    input_tensors = [torch_to_pb_tensor(name, data) for name, data in inputs.items()]

    # 2. Define requested outputs
    requested_output_names = OUTPUT_NAMES.get(model_name, [])

    # 3. Create Request
    inference_request = pb_utils.InferenceRequest(
        model_name=model_name,
        requested_output_names=requested_output_names,
        inputs=input_tensors,
    )

    # 4. Execute
    inference_response = inference_request.exec()

    if inference_response.has_error():
        raise RuntimeError(
            f"Model {model_name} failed: {inference_response.error().message()}"
        )

    # 5. Extract outputs (PB Tensor -> DLPack -> Torch)
    outputs = {}
    for name in requested_output_names:
        pb_out = pb_utils.get_output_tensor_by_name(inference_response, name)
        if pb_out is not None:
            outputs[name] = pb_tensor_to_torch(pb_out)

    return outputs


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

        # Calculate scores using PyTorch (on GPU)
        presence_score = torch.sigmoid(batch_presence)
        scores = torch.sigmoid(batch_logits) * presence_score

        # Filter by confidence threshold
        keep = scores > conf_threshold

        if keep.sum() == 0:
            all_masks.append(np.array([], dtype=bool))
            all_boxes.append(np.array([], dtype=np.float32))
            all_scores.append(np.array([], dtype=np.float32))
            continue

        # Move filtered tensors to CPU for resizing
        filtered_masks_torch = batch_masks[keep]  # Still on GPU
        filtered_boxes_torch = batch_boxes[keep]  # Still on GPU
        filtered_scores_torch = scores[keep]  # Still on GPU

        # Resize masks to original size (need CPU for cv2)
        filtered_masks = []
        for idx in range(keep.sum().item()):
            mask_288 = filtered_masks_torch[idx].cpu().numpy()  # [288, 288]
            # Resize to original size
            mask_resized = torch.nn.functional.interpolate(
                torch.from_numpy(mask_288[np.newaxis, np.newaxis, ...]).float(),
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            filtered_masks.append(mask_resized.cpu().numpy() > 0.5)

            # Scale boxes from normalized [0,1] to pixel coordinates (on GPU, then CPU)
            filtered_boxes = filtered_boxes_torch.clamp(0, 1)
            filtered_boxes[[0, 2]] *= orig_w
            filtered_boxes[[1, 3]] *= orig_h

        all_masks.append(filtered_masks)
        all_boxes.append(filtered_boxes.cpu())
        all_scores.append(filtered_scores_torch.cpu())

    # Create output tensors
    # Note: Triton doesn't support variable-length arrays directly,
    # so we'll return flattened arrays with metadata
    output_tensors = [
        pb_utils.Tensor("masks", np.array(all_masks)),
        pb_utils.Tensor("boxes", np.array(all_boxes)),
        pb_utils.Tensor("scores", np.array(all_scores)),
        pb_utils.Tensor(
            "num_detections",
            np.array([len(m) for m in all_masks], dtype=np.int32),
        ),
    ]
    return output_tensors
