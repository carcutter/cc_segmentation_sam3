"""
SAM3 ONNX Export Script (Dynamic Batch + TensorRT Compatible)
"""

import argparse
from pathlib import Path

import torch
from transformers.models.sam3.modeling_sam3 import Sam3Model

from model import (
    DecoderWrapper,
    GeometryEncoderWrapper,
    TextEncoderWrapper,
    VisionEncoderWrapper,
)


def export_vision_encoder(model: Sam3Model, output_dir: Path, device: str = "cuda"):
    print("Exporting Vision Encoder...")
    wrapper = VisionEncoderWrapper(model, device=device).to(device).eval()

    torch.onnx.export(
        wrapper,
        (torch.randn(1, 3, 1008, 1008, device=device),),
        str(output_dir / "vision-encoder.onnx"),
        input_names=["images"],
        output_names=["fpn_feat_0", "fpn_feat_1", "fpn_feat_2", "fpn_pos_2"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
        dynamic_axes={
            "images": {0: "batch"},
            "fpn_feat_0": {0: "batch"},
            "fpn_feat_1": {0: "batch"},
            "fpn_feat_2": {0: "batch"},
            "fpn_pos_2": {0: "batch"},
        },
    )
    print(f"  ✓ Saved: {output_dir / 'vision-encoder.onnx'}")


def export_text_encoder(model: Sam3Model, output_dir: Path, device: str = "cuda"):
    print("Exporting Text Encoder...")
    wrapper = TextEncoderWrapper(model).to(device).eval()

    torch.onnx.export(
        wrapper,
        (
            torch.randint(0, 49408, (1, 32), device=device),
            torch.ones(1, 32, dtype=torch.long, device=device),
        ),
        str(output_dir / "text-encoder.onnx"),
        input_names=["input_ids", "attention_mask"],
        output_names=["text_features", "text_mask"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
        dynamic_axes={
            "input_ids": {0: "batch"},
            "attention_mask": {0: "batch"},
            "text_features": {0: "batch"},
            "text_mask": {0: "batch"},
        },
    )
    print(f"  ✓ Saved: {output_dir / 'text-encoder.onnx'}")


def export_geometry_encoder(model: Sam3Model, output_dir: Path, device: str = "cuda"):
    print("Exporting Geometry Encoder...")
    wrapper = GeometryEncoderWrapper(model).to(device).eval()

    torch.onnx.export(
        wrapper,
        (
            torch.rand(1, 5, 4, device=device),
            torch.ones(1, 5, dtype=torch.long, device=device),
            torch.randn(1, 256, 72, 72, device=device),
            torch.randn(1, 256, 72, 72, device=device),
        ),
        str(output_dir / "geometry-encoder.onnx"),
        input_names=["input_boxes", "input_boxes_labels", "fpn_feat_2", "fpn_pos_2"],
        output_names=["geometry_features", "geometry_mask"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
        dynamic_axes={
            "input_boxes": {0: "batch", 1: "num_boxes"},
            "input_boxes_labels": {0: "batch", 1: "num_boxes"},
            "fpn_feat_2": {0: "batch"},
            "fpn_pos_2": {0: "batch"},
            "geometry_features": {0: "batch", 1: "num_prompts"},
            "geometry_mask": {0: "batch", 1: "num_prompts"},
        },
    )
    print(f"  ✓ Saved: {output_dir / 'geometry-encoder.onnx'}")


def export_decoder(model: Sam3Model, output_dir: Path, device: str = "cuda"):
    print("Exporting Decoder...")
    wrapper = DecoderWrapper(model).to(device).eval()

    torch.onnx.export(
        wrapper,
        (
            torch.randn(1, 256, 288, 288, device=device),
            torch.randn(1, 256, 144, 144, device=device),
            torch.randn(1, 256, 72, 72, device=device),
            torch.randn(1, 256, 72, 72, device=device),
            torch.randn(1, 32, 256, device=device),
            torch.ones(1, 32, dtype=torch.bool, device=device),
        ),
        str(output_dir / "decoder.onnx"),
        input_names=[
            "fpn_feat_0",
            "fpn_feat_1",
            "fpn_feat_2",
            "fpn_pos_2",
            "prompt_features",
            "prompt_mask",
        ],
        output_names=["pred_masks", "pred_boxes", "pred_logits", "presence_logits"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
        dynamic_axes={
            **{f"fpn_feat_{i}": {0: "batch"} for i in range(3)},
            "fpn_pos_2": {0: "batch"},
            "prompt_features": {0: "batch", 1: "prompt_len"},
            "prompt_mask": {0: "batch", 1: "prompt_len"},
            "pred_masks": {0: "batch"},
            "pred_boxes": {0: "batch"},
            "pred_logits": {0: "batch"},
            "presence_logits": {0: "batch"},
        },
    )
    print(f"  ✓ Saved: {output_dir / 'decoder.onnx'}")


def main():
    parser = argparse.ArgumentParser(
        description="Export SAM3 model to ONNX format with dynamic batch support"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="facebook/sam3",
        help="Path to SAM3 model directory",
    )
    parser.add_argument(
        "--output-dir", type=str, default="onnx-models", help="Output directory"
    )
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading SAM3 from {args.model_path}...")
    model = Sam3Model.from_pretrained(args.model_path).to(args.device).eval()
    print("  ✓ Model loaded\n")

    with torch.no_grad():
        export_vision_encoder(model, output_dir, args.device)
        export_text_encoder(model, output_dir, args.device)
        export_geometry_encoder(model, output_dir, args.device)
        export_decoder(model, output_dir, args.device)
    print(f"\n✓ Export complete! Models saved to: {output_dir}")


if __name__ == "__main__":
    main()
