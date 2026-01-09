"""
Triton Python Backend for SAM3 Inference Pipeline
Orchestrates vision-encoder, text-encoder, geometry-encoder, and decoder models
"""

import json
import logging
import os

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from tokenizers import Tokenizer

from utils import call_model, postprocess, preprocess_image, xywh_to_cxcywh_normalized

TARGET_SIZE = 1008
PAD_TOKEN_ID = 49407
MAX_TEXT_LENGTH = 32
DEVICE = "cuda"


class TritonPythonModel:
    """
    Triton Python Backend Model for SAM3 Inference Pipeline
    """

    def initialize(self, args):
        """
        Initialize the model.
        Called once when the model is being loaded.
        """
        self.model_config = json.loads(args["model_config"])
        model_version = args["model_version"]
        model_repository = args["model_repository"]
        model_name = args["model_name"]

        self.logger = logging.getLogger(__name__)
        self.logger.warning(f"Loaded model {model_name}, version: {model_version}")
        self.logger.warning(f"Model repository: {model_repository}")
        # Get model names from config
        self.vision_encoder_name = "vision-encoder"
        self.text_encoder_name = "text-encoder"
        self.geometry_encoder_name = "geometry-encoder"
        self.decoder_name = "decoder"
        self.logger.warning(
            f"Loading tokenizer from {model_repository}/{model_version}/tokenizer.json"
        )

        tokenizer_path = os.path.join(model_repository, model_version, "tokenizer.json")

        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_padding(length=MAX_TEXT_LENGTH, pad_id=PAD_TOKEN_ID)
        self.tokenizer.enable_truncation(max_length=MAX_TEXT_LENGTH)

        self.logger.info(f"Loaded tokenizer from {tokenizer_path}")
        self.logger.info("SAM3 Pipeline model initialized")

    def encode_text(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        encoded = self.tokenizer.encode(text)
        input_ids = torch.tensor([encoded.ids], dtype=torch.int64, device=DEVICE)
        attention_mask = torch.tensor(
            [encoded.attention_mask], dtype=torch.int64, device=DEVICE
        )

        return input_ids, attention_mask

    def execute(self, requests):
        responses = []

        for request in requests:
            try:
                # Extract inputs
                image_input = pb_utils.get_input_tensor_by_name(request, "image")
                text_input = pb_utils.get_input_tensor_by_name(request, "text")
                boxes_input = pb_utils.get_input_tensor_by_name(request, "boxes")
                box_labels_input = pb_utils.get_input_tensor_by_name(
                    request, "box_labels"
                )
                conf_threshold_input = pb_utils.get_input_tensor_by_name(
                    request, "conf_threshold"
                )

                # Get image bytes
                image_bytes = image_input.as_numpy().tobytes()

                # Get text prompt
                text = ""
                if text_input is not None:
                    text_bytes = text_input.as_numpy().tobytes()
                    text = text_bytes.decode("utf-8").strip("\x00")

                # Get boxes (optional)
                boxes = None
                box_labels = None
                if boxes_input is not None:
                    boxes_array = boxes_input.as_numpy()
                    boxes = boxes_array.tolist() if boxes_array.size > 0 else None

                if box_labels_input is not None:
                    box_labels_array = box_labels_input.as_numpy()
                    box_labels = (
                        box_labels_array.tolist() if box_labels_array.size > 0 else None
                    )

                # Get confidence threshold
                conf_threshold = 0.5
                if conf_threshold_input is not None:
                    conf_threshold = float(conf_threshold_input.as_numpy()[0])

                # Preprocess image
                pixel_values, orig_size = preprocess_image(image_bytes)
                orig_h, orig_w = orig_size

                print("Finished preprocess image")

                # Convert pixel_values to torch tensor on GPU
                pixel_values_torch = torch.from_numpy(pixel_values).to(DEVICE)

                # Step 1: Vision Encoder (returns PyTorch tensors on GPU)
                vision_outputs = call_model(
                    self.vision_encoder_name,
                    {"images": pixel_values_torch},
                )
                fpn_feat_0 = vision_outputs["fpn_feat_0"]  # Already torch.Tensor on GPU
                fpn_feat_1 = vision_outputs["fpn_feat_1"]
                fpn_feat_2 = vision_outputs["fpn_feat_2"]
                fpn_pos_2 = vision_outputs["fpn_pos_2"]

                # Step 2: Text Encoder
                input_ids, attention_mask = self.encode_text(text)

                text_outputs = call_model(
                    self.text_encoder_name,
                    {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                    },
                )
                text_features = text_outputs["text_features"]  # torch.Tensor on GPU
                text_mask = text_outputs["text_mask"]  # torch.Tensor on GPU

                # Step 3: Geometry Encoder (if boxes provided)
                if boxes and len(boxes) > 0:
                    # Convert boxes to normalized cxcywh format
                    boxes_cxcywh = xywh_to_cxcywh_normalized(boxes, orig_w, orig_h)
                    boxes_array = torch.from_numpy(boxes_cxcywh.reshape(1, -1, 4)).to(
                        DEVICE
                    )

                    if box_labels:
                        labels_array = torch.tensor(
                            box_labels, dtype=torch.int64, device=DEVICE
                        ).reshape(1, -1)
                    else:
                        labels_array = torch.ones(
                            (1, len(boxes)), dtype=torch.int64, device=DEVICE
                        )

                    geom_outputs = call_model(
                        self.geometry_encoder_name,
                        {
                            "input_boxes": boxes_array,
                            "input_boxes_labels": labels_array,
                            "fpn_feat_2": fpn_feat_2,
                            "fpn_pos_2": fpn_pos_2,
                        },
                    )
                    geometry_features = geom_outputs[
                        "geometry_features"
                    ]  # torch.Tensor on GPU
                    geometry_mask = geom_outputs["geometry_mask"]  # torch.Tensor on GPU

                    # Concatenate text and geometry features (PyTorch operation on GPU)
                    prompt_features = torch.cat(
                        [text_features, geometry_features], dim=1
                    )
                    prompt_mask = torch.cat([text_mask, geometry_mask], dim=1)
                else:
                    # No boxes: use text features only
                    prompt_features = text_features
                    prompt_mask = text_mask

                # Step 4: Decoder
                decoder_outputs = call_model(
                    self.decoder_name,
                    {
                        "fpn_feat_0": fpn_feat_0,
                        "fpn_feat_1": fpn_feat_1,
                        "fpn_feat_2": fpn_feat_2,
                        "fpn_pos_2": fpn_pos_2,
                        "prompt_features": prompt_features,
                        "prompt_mask": prompt_mask.bool(),
                    },
                )

                output_tensors = postprocess(
                    decoder_outputs, orig_h, orig_w, conf_threshold
                )

                inference_response = pb_utils.InferenceResponse(
                    output_tensors=output_tensors
                )
                responses.append(inference_response)

            except Exception as e:
                self.logger.error(f"Error processing request: {e}")
                error_response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(f"Error: {str(e)}")
                )
                responses.append(error_response)

        return responses

    def finalize(self):
        """Called when the model is being unloaded."""
        self.logger.info("SAM3 Pipeline model finalized")
