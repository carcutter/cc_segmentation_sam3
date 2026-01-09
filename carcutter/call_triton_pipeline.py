#!/usr/bin/env python3
"""
Client script to call Triton Inference Server SAM3 Pipeline model
"""

import argparse
import io
import sys

import numpy as np
import tritonclient.http as httpclient
from PIL import Image

from utils import get_args, save_results


class TritonPipelineClient:
    """Client for calling SAM3 Pipeline model on Triton Server"""

    def __init__(self, url: str = "localhost:8000", model_name: str = "pipeline"):
        """
        Initialize Triton client.

        Args:
            url: Triton server URL (default: localhost:8000)
            model_name: Name of the pipeline model (default: "pipeline")
        """
        self.url = url
        self.model_name = model_name
        self.client = None

    def connect(self):
        """Connect to Triton server"""
        try:
            self.client = httpclient.InferenceServerClient(url=self.url)
            print(f"✓ Connected to Triton server at {self.url}")
        except Exception as e:
            print(f"✗ Failed to connect to Triton server: {e}")
            raise

    def is_ready(self) -> bool:
        """Check if the model is ready"""
        if self.client is None:
            self.connect()

        try:
            return self.client.is_model_ready(self.model_name)
        except Exception as e:
            print(f"✗ Error checking model readiness: {e}")
            return False

    def infer(
        self,
        image_path: str,
        text: str = None,
        boxes: list = None,
        box_labels: list = None,
        conf_threshold: float = 0.3,
    ) -> dict:
        """
        Run inference on the pipeline model.

        Args:
            image_path: Path to input image
            text: Text prompt (optional)
            boxes: List of boxes in xywh format [[x,y,w,h], ...] (optional)
            box_labels: List of box labels [1, 0, ...] where 1=positive, 0=negative (optional)
            conf_threshold: Confidence threshold (default: 0.3)

        Returns:
            Dictionary with keys: masks, boxes, scores, num_detections
        """
        if self.client is None:
            self.connect()

        if not self.is_ready():
            raise RuntimeError(f"Model '{self.model_name}' is not ready")

        # Load and encode image
        print(f"\nLoading image: {image_path}")
        image = Image.open(image_path).convert("RGB")
        orig_size = image.size  # (width, height)
        print(f"  Original size: {orig_size[0]}x{orig_size[1]}")

        # Convert image to bytes (JPEG)
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="JPEG", quality=95)
        image_bytes = image_bytes.getvalue()
        print(f"  Image size: {len(image_bytes)} bytes")

        # Prepare inputs
        inputs = []

        # Image input (required)
        # For max_batch_size > 0, Triton expects batch dimension as first dimension
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        # Reshape to [1, len] to indicate batch size of 1
        image_array = image_array.reshape(1, -1)
        image_input = httpclient.InferInput("image", [1, image_array.shape[1]], "UINT8")
        image_input.set_data_from_numpy(image_array)
        inputs.append(image_input)

        # Text input (optional)
        if text:
            text_bytes = text.encode("utf-8")
            text_array = np.frombuffer(text_bytes, dtype=np.uint8)
            # Reshape to [1, len] to indicate batch size of 1
            text_array = text_array.reshape(1, -1)
            text_input = httpclient.InferInput(
                "text", [1, text_array.shape[1]], "UINT8"
            )
            text_input.set_data_from_numpy(text_array)
            inputs.append(text_input)
            print(f"  Text prompt: '{text}'")

        # Boxes input (optional)
        if boxes and len(boxes) > 0:
            boxes_array = np.array(boxes, dtype=np.float32)
            # Ensure correct shape: [batch=1, num_boxes, 4]
            if len(boxes_array.shape) == 1:
                boxes_array = boxes_array.reshape(1, 1, -1)
            elif len(boxes_array.shape) == 2:
                # Add batch dimension: [num_boxes, 4] -> [1, num_boxes, 4]
                boxes_array = boxes_array.reshape(
                    1, boxes_array.shape[0], boxes_array.shape[1]
                )
            boxes_input = httpclient.InferInput("boxes", boxes_array.shape, "FP32")
            boxes_input.set_data_from_numpy(boxes_array)
            inputs.append(boxes_input)
            print(f"  Boxes: {len(boxes)} box(es)")

            # Box labels input (optional, but recommended when boxes are provided)
            if box_labels:
                labels_array = np.array(box_labels, dtype=np.int64)
                # Ensure shape: [batch=1, num_boxes]
                if len(labels_array.shape) == 0:
                    labels_array = labels_array.reshape(1, 1)
                elif len(labels_array.shape) == 1:
                    labels_array = labels_array.reshape(1, -1)
                labels_input = httpclient.InferInput(
                    "box_labels", labels_array.shape, "INT64"
                )
                labels_input.set_data_from_numpy(labels_array)
                inputs.append(labels_input)
                print(f"  Box labels: {box_labels}")

        # Confidence threshold input (optional)
        # Shape: [batch=1, 1] for consistency
        conf_array = np.array([[conf_threshold]], dtype=np.float32)
        conf_input = httpclient.InferInput("conf_threshold", [1, 1], "FP32")
        conf_input.set_data_from_numpy(conf_array)
        inputs.append(conf_input)
        print(f"  Confidence threshold: {conf_threshold}")

        # Prepare outputs
        outputs = [
            httpclient.InferRequestedOutput("masks"),
            httpclient.InferRequestedOutput("boxes"),
            httpclient.InferRequestedOutput("scores"),
            httpclient.InferRequestedOutput("num_detections"),
        ]

        # Run inference
        print("\nRunning inference...")
        # try:
        results = self.client.infer(self.model_name, inputs, outputs=outputs)

        # Extract results
        num_detections = results.as_numpy("num_detections")
        scores = results.as_numpy("scores")
        boxes = results.as_numpy("boxes")
        masks = results.as_numpy("masks")

        print("✓ Inference completed")
        print(f"  Detections: {num_detections[0]}")

        return {
            "masks": masks,
            "boxes": boxes,
            "scores": scores,
            "num_detections": num_detections,
            "original_size": orig_size,
        }


def parse_box_string(box_str: str) -> list:
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
    parser = argparse.ArgumentParser(
        description="Call Triton Inference Server SAM3 Pipeline model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Text prompt only
  python call_triton_pipeline.py --image image.jpg --text "car"

  # Box prompts only
  python call_triton_pipeline.py --image image.jpg --boxes "100,100,200,200"

  # Text + boxes
  python call_triton_pipeline.py --image image.jpg --text "car" --boxes "100,100,200,200" --box-labels "1"

  # Custom confidence threshold
  python call_triton_pipeline.py --image image.jpg --text "car" --conf 0.5

  # Save visualization
  python call_triton_pipeline.py --image image.jpg --text "car" --output result.png
        """,
    )

    args = get_args()

    # Validate inputs
    if not args.text and not args.boxes:
        parser.error("Please provide either --text or --boxes (or both)")

    # Parse boxes
    boxes = None
    if args.boxes:
        boxes = parse_box_string(args.boxes)
        if not boxes:
            parser.error(f"Invalid box format: {args.boxes}")

    # Parse box labels
    box_labels = None
    if args.box_labels:
        box_labels = [int(x.strip()) for x in args.box_labels.split(",")]
        if boxes and len(box_labels) != len(boxes):
            parser.error(
                f"Number of box labels ({len(box_labels)}) must match number of boxes ({len(boxes)})"
            )

    # Create client and run inference
    try:
        client = TritonPipelineClient(url=args.url, model_name=args.model)

        results = client.infer(
            image_path=args.image,
            text=args.text,
            boxes=boxes,
            box_labels=box_labels,
            conf_threshold=args.conf,
        )

        save_results(args.image, results, args.output)
        return 0

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
