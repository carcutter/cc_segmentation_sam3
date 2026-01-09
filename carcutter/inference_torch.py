"""
SAM3 PyTorch Inference Script
"""

from typing import Optional

import cv2
import numpy as np
import torch
from tokenizers import Tokenizer
from transformers import Sam3Model

from model import (
    DecoderWrapper,
    GeometryEncoderWrapper,
    TextEncoderWrapper,
    VisionEncoderWrapper,
)
from utils import (
    get_args,
    parse_box_prompts,
    visualize_results,
    xywh_to_cxcywh_normalized,
)

TARGET_SIZE = 1008


class Sam3PyTorchInference:
    """SAM3 PyTorch Inference Engine"""

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        device: str = "cuda",
    ):
        print("Loading PyTorch model...")
        self.device = torch.device(
            device if torch.cuda.is_available() and device == "cuda" else "cpu"
        )

        # Load model
        self.model = Sam3Model.from_pretrained(model_path).to(self.device).eval()

        # Create wrapper modules
        self.vision_encoder = (
            VisionEncoderWrapper(self.model, device=self.device).to(self.device).eval()
        )
        self.text_encoder = TextEncoderWrapper(self.model).to(self.device).eval()
        self.geometry_encoder = (
            GeometryEncoderWrapper(self.model).to(self.device).eval()
        )
        self.decoder = DecoderWrapper(self.model).to(self.device).eval()

        # Load tokenizer
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_padding(length=32, pad_id=49407)
        self.tokenizer.enable_truncation(max_length=32)
        print("  ✓ All models loaded")

    def preprocess_image(
        self, image: np.ndarray
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        """Preprocess: resize to target size and normalize"""
        orig_size = image.shape[:2]  # (h, w)
        from PIL import Image as PILImage

        pil_image = PILImage.fromarray(image)
        resized = np.array(
            pil_image.resize((TARGET_SIZE, TARGET_SIZE), PILImage.BILINEAR)
        )
        normalized = resized.astype(np.float32) / 127.5 - 1.0  # [0,255] -> [-1,1]
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1)[np.newaxis]).to(
            self.device
        )  # NCHW
        return tensor, orig_size

    def encode_image(self, pixel_values: torch.Tensor) -> dict:
        """Encode image using vision encoder"""
        with torch.no_grad():
            outputs = self.vision_encoder(pixel_values)
        return {
            "fpn_feat_0": outputs[0].cpu().numpy(),  # [B, 256, 288, 288]
            "fpn_feat_1": outputs[1].cpu().numpy(),  # [B, 256, 144, 144]
            "fpn_feat_2": outputs[2].cpu().numpy(),  # [B, 256, 72, 72]
            "fpn_pos_2": outputs[3].cpu().numpy(),  # [B, 256, 72, 72]
        }

    def encode_text(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        """Encode text prompt"""
        self.tokenizer.enable_padding(pad_id=49407, length=32)
        self.tokenizer.enable_truncation(max_length=32)

        encoded = self.tokenizer.encode(text)
        input_ids = torch.tensor([encoded.ids], dtype=torch.int64, device=self.device)
        attention_mask = torch.tensor(
            [encoded.attention_mask], dtype=torch.int64, device=self.device
        )

        with torch.no_grad():
            outputs = self.text_encoder(input_ids, attention_mask)
        return outputs[0].cpu().numpy(), outputs[1].cpu().numpy()

    def encode_boxes(
        self,
        boxes: np.ndarray,
        labels: np.ndarray,
        fpn_feat: np.ndarray,
        fpn_pos: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode box prompts"""
        boxes_tensor = torch.from_numpy(boxes.astype(np.float32)).to(self.device)
        labels_tensor = torch.from_numpy(labels.astype(np.int64)).to(self.device)
        fpn_feat_tensor = torch.from_numpy(fpn_feat).to(self.device)
        fpn_pos_tensor = torch.from_numpy(fpn_pos).to(self.device)

        with torch.no_grad():
            outputs = self.geometry_encoder(
                boxes_tensor,
                labels_tensor,
                fpn_feat_tensor,
                fpn_pos_tensor,
            )
        return outputs[0].cpu().numpy(), outputs[1].cpu().numpy()

    def decode(
        self,
        vision_features: dict,
        prompt_features: np.ndarray,
        prompt_mask: np.ndarray,
    ) -> dict:
        """Decode features to generate masks"""
        fpn_feat_0 = torch.from_numpy(vision_features["fpn_feat_0"]).to(self.device)
        fpn_feat_1 = torch.from_numpy(vision_features["fpn_feat_1"]).to(self.device)
        fpn_feat_2 = torch.from_numpy(vision_features["fpn_feat_2"]).to(self.device)
        fpn_pos_2 = torch.from_numpy(vision_features["fpn_pos_2"]).to(self.device)
        prompt_features_tensor = torch.from_numpy(prompt_features).to(self.device)
        prompt_mask_tensor = torch.from_numpy(prompt_mask).to(self.device)

        with torch.no_grad():
            outputs = self.decoder(
                fpn_feat_0,
                fpn_feat_1,
                fpn_feat_2,
                fpn_pos_2,
                prompt_features_tensor,
                prompt_mask_tensor,
            )
        return {
            "pred_masks": outputs[0].cpu().numpy(),
            "pred_boxes": outputs[1].cpu().numpy(),
            "pred_logits": outputs[2].cpu().numpy(),
            "presence_logits": outputs[3].cpu().numpy(),
        }

    def predict(
        self,
        image: np.ndarray,
        text: Optional[str] = None,
        boxes: Optional[list] = None,
        box_labels: Optional[list] = None,
        conf_threshold: float = 0.3,
    ) -> dict:
        """Unified prediction with text and/or box prompts

        Args:
            image: RGB image [H, W, 3]
            text: Text prompt (optional)
            boxes: Box prompts [[x,y,w,h], ...] in xywh pixel format (optional)
            box_labels: Box labels [1, 0, ...] 1=pos, 0=neg (optional)
            conf_threshold: Confidence threshold
        """
        pixel_values, orig_size = self.preprocess_image(image)
        vision_features = self.encode_image(pixel_values)
        h, w = orig_size

        # Encode text
        if text:
            text_features, text_mask = self.encode_text(text)
        else:
            # No text: use padding tokens (length=32)
            pad_ids = torch.full((1, 32), 49407, dtype=torch.int64, device=self.device)
            pad_mask = torch.zeros((1, 32), dtype=torch.int64, device=self.device)
            pad_mask[0, 0] = 1  # at least one valid token
            with torch.no_grad():
                outputs = self.text_encoder(pad_ids, pad_mask)
            text_features, text_mask = (
                outputs[0].cpu().numpy(),
                outputs[1].cpu().numpy(),
            )

        # Encode boxes
        if boxes and len(boxes) > 0:
            boxes_cxcywh = xywh_to_cxcywh_normalized(boxes, w, h)
            boxes_array = boxes_cxcywh.reshape(1, -1, 4)
            if box_labels:
                labels_array = np.array(box_labels, dtype=np.int64).reshape(1, -1)
            else:
                labels_array = np.ones((1, len(boxes)), dtype=np.int64)
            geom_features, geom_mask = self.encode_boxes(
                boxes_array,
                labels_array,
                vision_features["fpn_feat_2"],
                vision_features["fpn_pos_2"],
            )
            # Concatenate text and geometry features
            prompt_features = np.concatenate([text_features, geom_features], axis=1)
            prompt_mask = np.concatenate([text_mask, geom_mask], axis=1)
        else:
            # No boxes: use text features only
            prompt_features = text_features
            prompt_mask = text_mask

        outputs = self.decode(vision_features, prompt_features, prompt_mask)
        return self._postprocess(outputs, orig_size, conf_threshold, boxes)

    def _postprocess(
        self,
        outputs: dict,
        orig_size: tuple[int, int],
        conf_threshold: float,
        input_boxes: Optional[list] = None,
    ) -> dict:
        """Post-process model outputs"""
        pred_masks = outputs["pred_masks"][0]
        pred_boxes = outputs["pred_boxes"][0]
        pred_logits = outputs["pred_logits"][0]
        presence_logits = outputs["presence_logits"][0, 0]

        presence_score = 1 / (1 + np.exp(-presence_logits))
        scores = (1 / (1 + np.exp(-pred_logits))) * presence_score
        keep = scores > conf_threshold

        h, w = orig_size
        # Resize masks: 288x288 -> original size
        masks = []
        for m in pred_masks[keep]:
            mask_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
            masks.append(mask_resized > 0)
        # Scale boxes from normalized [0,1] to pixel coordinates
        boxes = pred_boxes[keep].copy()
        boxes[:, [0, 2]] *= w
        boxes[:, [1, 3]] *= h
        boxes = np.clip(boxes, 0, [[w, h, w, h]])

        return {
            "masks": masks,
            "boxes": boxes,
            "scores": scores[keep],
            "orig_size": orig_size,
            "input_boxes": input_boxes,
        }


def main():
    args = get_args()

    # Load model
    engine = Sam3PyTorchInference(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer,
        device=args.device,
    )

    # Load image
    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        raise ValueError(f"Cannot load image: {args.image}")
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    print(f"\nProcessing: {args.image} ({image.shape[1]}x{image.shape[0]})")

    # Parse prompts
    boxes, box_labels = None, None
    if args.boxes:
        boxes, box_labels = parse_box_prompts(args.boxes)
        print(f"  Box prompts: {len(boxes)} boxes, labels={box_labels}")
    if args.text:
        print(f"  Text prompt: '{args.text}'")

    # Run inference
    results = engine.predict(
        image,
        text=args.text,
        boxes=boxes,
        box_labels=box_labels,
        conf_threshold=args.conf,
    )

    print(f"  Found {len(results['masks'])} objects")

    # Visualize
    visualize_results(image_bgr, results, args.output)


if __name__ == "__main__":
    main()
