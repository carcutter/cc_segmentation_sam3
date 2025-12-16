#!/usr/bin/env python3
"""
Client script to call individual Triton Inference Server models
Follows the same logic as model.py for preprocessing and model calls
"""

import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import tritonclient.http as httpclient
from PIL import Image
from tokenizers import Tokenizer
from tritonclient.utils import np_to_triton_dtype

from utils import get_args, postprocess, preprocess_image, visualize_results

TARGET_SIZE = 1008
PAD_TOKEN_ID = 49407
MAX_TEXT_LENGTH = 32
TRITON_URL = "localhost:8000"


class TritonModelClient:
    """Client for calling individual Triton models"""

    def __init__(self, url: str = TRITON_URL):
        """
        Initialize Triton client.

        Args:
            url: Triton server URL (default: localhost:8000)
        """
        self.url = url
        self.client = None
        self.tokenizer = None

    def connect(self):
        """Connect to Triton server"""
        try:
            self.client = httpclient.InferenceServerClient(url=self.url)
            print(f"✓ Connected to Triton server at {self.url}")
        except Exception as e:
            print(f"✗ Failed to connect to Triton server: {e}")
            raise

    def is_model_ready(
        self, model_name: str, model_version: Optional[str] = None
    ) -> bool:
        """Check if a model is ready"""
        if self.client is None:
            self.connect()

        try:
            return self.client.is_model_ready(model_name, model_version)
        except Exception as e:
            print(f"✗ Error checking model readiness: {e}")
            return False

    def call_triton_model(
        self,
        model_name: str,
        inputs: Dict[str, np.ndarray],
        output_names: List[str],
        model_version: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Generic function to call any Triton model.

        Args:
            model_name: Name of the model to call
            inputs: Dictionary mapping input names to numpy arrays
            output_names: List of output names to request
            model_version: Model version (optional, None means latest)

        Returns:
            Dictionary mapping output names to numpy arrays
        """
        if self.client is None:
            self.connect()

        if not self.is_model_ready(model_name, model_version):
            raise RuntimeError(
                f"Model '{model_name}' (version: {model_version}) is not ready"
            )

        # Prepare input tensors
        input_tensors = []
        for name, data in inputs.items():
            # Ensure data is a numpy array
            if not isinstance(data, np.ndarray):
                data = np.array(data)

            # Get shape and dtype
            shape = data.shape
            dtype = data.dtype

            # Map numpy dtype to Triton dtype string
            dtype_str = np_to_triton_dtype(dtype)

            # Create InferInput
            infer_input = httpclient.InferInput(name, shape, dtype_str)
            infer_input.set_data_from_numpy(data)
            input_tensors.append(infer_input)

        # Prepare output tensors
        output_tensors = [
            httpclient.InferRequestedOutput(name) for name in output_names
        ]

        # Run inference
        results = self.client.infer(
            model_name,
            input_tensors,
            outputs=output_tensors,
            model_version=model_version,
        )

        # Extract outputs
        outputs = {}
        for name in output_names:
            outputs[name] = results.as_numpy(name)

        return outputs

    def encode_text(
        self, text: str, tokenizer_path: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        tokenizer_path = "model_repository/pipeline/1/tokenizer.json"
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_padding(length=MAX_TEXT_LENGTH, pad_id=PAD_TOKEN_ID)
        self.tokenizer.enable_truncation(max_length=MAX_TEXT_LENGTH)

        encoded = self.tokenizer.encode(text)
        input_ids = np.array([encoded.ids], dtype=np.int64)
        attention_mask = np.array([encoded.attention_mask], dtype=np.int64)

        return input_ids, attention_mask

    def call_vision_encoder(
        self, image_path: str, model_version: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        Call vision-encoder model.

        Args:
            image_path: Path to input image
            model_version: Model version (optional)

        Returns:
            Dictionary with keys: fpn_feat_0, fpn_feat_1, fpn_feat_2, fpn_pos_2
        """
        # Load and preprocess image
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        pixel_values, orig_size = preprocess_image(image_bytes)

        # Call model
        outputs = self.call_triton_model(
            model_name="vision-encoder",
            inputs={"images": pixel_values},
            output_names=["fpn_feat_0", "fpn_feat_1", "fpn_feat_2", "fpn_pos_2"],
            model_version=model_version,
        )

        outputs["original_size"] = orig_size
        return outputs

    def call_text_encoder(
        self,
        text: str,
        tokenizer_path: Optional[str] = None,
        model_version: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Call text-encoder model.

        Args:
            text: Text prompt
            tokenizer_path: Path to tokenizer.json (optional)
            model_version: Model version (optional)

        Returns:
            Dictionary with keys: text_features, text_mask
        """
        input_ids, attention_mask = self.encode_text(text, tokenizer_path)

        # Call model
        outputs = self.call_triton_model(
            model_name="text-encoder",
            inputs={"input_ids": input_ids, "attention_mask": attention_mask},
            output_names=["text_features", "text_mask"],
            model_version=model_version,
        )

        return outputs

    def call_geometry_encoder(
        self,
        boxes: List[List[float]],
        fpn_feat_2: np.ndarray,
        fpn_pos_2: np.ndarray,
        box_labels: Optional[List[int]] = None,
        img_w: Optional[int] = None,
        img_h: Optional[int] = None,
        model_version: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Call geometry-encoder model.

        Args:
            boxes: List of boxes in xywh format [[x,y,w,h], ...]
            fpn_feat_2: FPN feature from vision encoder (shape: [1, 256, 72, 72] or [256, 72, 72])
            fpn_pos_2: FPN position from vision encoder (shape: [1, 256, 72, 72] or [256, 72, 72])
            box_labels: List of box labels [1, 0, ...] where 1=positive, 0=negative (optional)
            img_w: Image width (required if boxes are in pixel coordinates)
            img_h: Image height (required if boxes are in pixel coordinates)
            model_version: Model version (optional)

        Returns:
            Dictionary with keys: geometry_features, geometry_mask
        """
        # Convert boxes to normalized cxcywh if needed
        if img_w is not None and img_h is not None:
            boxes_cxcywh = self.xywh_to_cxcywh_normalized(boxes, img_w, img_h)
        else:
            # Assume boxes are already in normalized cxcywh format
            boxes_cxcywh = np.array(boxes, dtype=np.float32)

        # Ensure correct shape: [batch=1, num_boxes, 4]
        if len(boxes_cxcywh.shape) == 2:
            boxes_cxcywh = boxes_cxcywh.reshape(1, -1, 4)
        elif len(boxes_cxcywh.shape) == 1:
            boxes_cxcywh = boxes_cxcywh.reshape(1, 1, -1)

        # Prepare box labels
        if box_labels is None:
            labels_array = np.ones((1, len(boxes)), dtype=np.int64)
        else:
            labels_array = np.array(box_labels, dtype=np.int64).reshape(1, -1)

        # Ensure fpn_feat_2 and fpn_pos_2 have correct shape (remove batch dim if present)
        if len(fpn_feat_2.shape) == 4:
            fpn_feat_2 = fpn_feat_2[0]  # Remove batch dimension
        if len(fpn_pos_2.shape) == 4:
            fpn_pos_2 = fpn_pos_2[0]  # Remove batch dimension

        # Call model
        outputs = self.call_triton_model(
            model_name="geometry-encoder",
            inputs={
                "input_boxes": boxes_cxcywh,
                "input_boxes_labels": labels_array,
                "fpn_feat_2": fpn_feat_2,
                "fpn_pos_2": fpn_pos_2,
            },
            output_names=["geometry_features", "geometry_mask"],
            model_version=model_version,
        )

        return outputs

    def call_decoder(
        self,
        fpn_feat_0: np.ndarray,
        fpn_feat_1: np.ndarray,
        fpn_feat_2: np.ndarray,
        fpn_pos_2: np.ndarray,
        prompt_features: np.ndarray,
        prompt_mask: np.ndarray,
        model_version: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Call decoder model.

        Args:
            fpn_feat_0: FPN feature 0 from vision encoder
            fpn_feat_1: FPN feature 1 from vision encoder
            fpn_feat_2: FPN feature 2 from vision encoder
            fpn_pos_2: FPN position from vision encoder
            prompt_features: Prompt features (text + geometry concatenated)
            prompt_mask: Prompt mask (text + geometry concatenated)
            model_version: Model version (optional)

        Returns:
            Dictionary with keys: pred_masks, pred_boxes, pred_logits, presence_logits
        """
        # Ensure prompt_mask is boolean
        if prompt_mask.dtype != np.bool_:
            prompt_mask = prompt_mask.astype(np.bool_)

        # Call model
        outputs = self.call_triton_model(
            model_name="decoder",
            inputs={
                "fpn_feat_0": fpn_feat_0,
                "fpn_feat_1": fpn_feat_1,
                "fpn_feat_2": fpn_feat_2,
                "fpn_pos_2": fpn_pos_2,
                "prompt_features": prompt_features,
                "prompt_mask": prompt_mask,
            },
            output_names=["pred_masks", "pred_boxes", "pred_logits", "presence_logits"],
            model_version=model_version,
        )

        return outputs


def parse_box_string(box_str: str) -> List[List[float]]:
    """Parse box string format: 'x1,y1,w1,h1;x2,y2,w2,h2' or 'x1,y1,w1,h1'"""
    boxes = []
    for part in box_str.split(";"):
        part = part.strip()
        if not part:
            continue
        coords = [float(x) for x in part.split(",")]
        if len(coords) == 4:
            boxes.append(coords)
    return boxes


def main():
    args = get_args()

    # Create client
    client = TritonModelClient(url=TRITON_URL)

    outputs = None

    # Get vision encoder outputs
    print("Getting vision encoder outputs...")
    vision_outputs = client.call_vision_encoder(args.image, "1")
    fpn_feat_0 = vision_outputs["fpn_feat_0"]
    fpn_feat_1 = vision_outputs["fpn_feat_1"]
    fpn_feat_2 = vision_outputs["fpn_feat_2"]
    fpn_pos_2 = vision_outputs["fpn_pos_2"]
    orig_h, orig_w = vision_outputs["original_size"]

    # Get text encoder outputs
    if args.text:
        print("Getting text encoder outputs...")
        text_outputs = client.call_text_encoder(args.text, args.tokenizer, "1")
        prompt_features = text_outputs["text_features"]
        prompt_mask = text_outputs["text_mask"]
    else:
        # Create empty text features
        print("Warning: No prompt provided, using empty text features...")
        prompt_features = np.zeros((1, MAX_TEXT_LENGTH, 256), dtype=np.float32)
        prompt_mask = np.zeros((1, MAX_TEXT_LENGTH), dtype=np.bool_)

    # Get geometry encoder outputs if boxes provided
    if args.boxes:
        print("Getting geometry encoder outputs...")
        boxes = parse_box_string(args.boxes)
        box_labels = None
        if args.box_labels:
            box_labels = [int(x.strip()) for x in args.box_labels.split(",")]

        geom_outputs = client.call_geometry_encoder(
            boxes, fpn_feat_2, fpn_pos_2, box_labels, orig_w, orig_h, "1"
        )

        # Concatenate text and geometry features
        prompt_features = np.concatenate(
            [prompt_features, geom_outputs["geometry_features"]], axis=1
        )
        prompt_mask = np.concatenate(
            [prompt_mask, geom_outputs["geometry_mask"]], axis=1
        )

    print("Calling decoder...")
    outputs = client.call_decoder(
        fpn_feat_0, fpn_feat_1, fpn_feat_2, fpn_pos_2, prompt_features, prompt_mask, "1"
    )

    results = postprocess(outputs, orig_h, orig_w, 0.5)

    visualize_results(args.image, results, args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
