#!/usr/bin/env python3
"""
Chain pseudo-GT analysis script.

Validates whether "chain" prompts from the base SAM3 model produce sensible
sub-segmentations of the trailer body region.

Workflow:
  1. Run run_inference.py with --prompt "chain" to produce chain predictions.
  2. Run this script to intersect those predictions with the GT body mask
     and generate per-image visualisations + a summary CSV.

The intersection (chain prediction ∩ GT body mask) is the candidate pseudo-GT
for chains. The visualisations let you judge whether it looks reasonable before
committing to using it as ground truth for model evaluation.

Usage:
    python carcutter/eval/analyze_chain_pseudogt.py \\
        --image-dir     /path/to/images \\
        --gt-mask-dir   /path/to/gt_color_masks \\
        --chain-pred-dir /path/to/chain_predictions \\
        --output-dir    /path/to/analysis_output

    # If the GT body colour is not red (255,0,0), override it:
    python carcutter/eval/analyze_chain_pseudogt.py \\
        ... \\
        --body-color "255,0,0"
"""

import argparse
import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def load_binary(path: Path) -> np.ndarray:
    """Load any image as a boolean mask (True where non-zero)."""
    img = np.array(Image.open(path).convert("L"))
    return img > 0


def extract_body_mask(gt_color: np.ndarray, body_rgb: tuple[int, int, int]) -> np.ndarray:
    """Return boolean mask of pixels matching body_rgb in the GT colour annotation."""
    r, g, b = body_rgb
    return (
        (gt_color[:, :, 0] == r) &
        (gt_color[:, :, 1] == g) &
        (gt_color[:, :, 2] == b)
    )


def make_visualization(
    image: np.ndarray,
    body_mask: np.ndarray,
    chain_pred: np.ndarray,
    pseudogt: np.ndarray,
    image_name: str,
    stats: dict,
) -> plt.Figure:
    """4-panel figure: original | GT body | chain prediction | intersection (pseudo-GT)."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(
        f"{image_name}  |  body: {stats['gt_body_px']:,}px  "
        f"chain pred: {stats['chain_pred_px']:,}px  "
        f"intersection: {stats['intersection_px']:,}px  "
        f"coverage: {stats['chain_coverage_pct']:.1f}%  "
        f"precision: {stats['chain_pred_precision_pct']:.1f}%",
        fontsize=9,
    )

    titles = ["Original", "GT body mask", "Chain prediction", "Intersection (pseudo-GT)"]
    overlays = [None, body_mask, chain_pred, pseudogt]
    colors = [None, (0.9, 0.2, 0.2), (0.2, 0.7, 0.2), (0.2, 0.5, 1.0)]

    for ax, title, mask, color in zip(axes, titles, overlays, colors):
        ax.imshow(image)
        if mask is not None and mask.any():
            overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
            overlay[mask] = (*color, 0.55)
            ax.imshow(overlay)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    return fig


def run_analysis(
    image_dir: str,
    gt_mask_dir: str,
    chain_pred_dir: str,
    output_dir: str,
    body_color: tuple[int, int, int],
) -> None:
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    image_files = sorted(
        p for p in Path(image_dir).iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_files:
        print(f"No images found in {image_dir}")
        return

    csv_path = os.path.join(output_dir, "chain_pseudogt_analysis.csv")
    fieldnames = [
        "image",
        "gt_body_px",
        "chain_pred_px",
        "intersection_px",
        "chain_coverage_pct",   # intersection / gt_body — how much of the body is chain
        "chain_pred_precision_pct",  # intersection / chain_pred — how much of chain pred is within body
        "gt_mask_found",
        "chain_pred_found",
    ]

    rows = []
    skipped = []

    print(f"Analysing {len(image_files)} images...")
    for img_path in image_files:
        stem = img_path.stem

        # Locate GT mask and chain prediction (try common suffixes)
        gt_path = _find_file(Path(gt_mask_dir), stem)
        chain_path = _find_file(Path(chain_pred_dir), stem)

        gt_found = gt_path is not None
        chain_found = chain_path is not None

        if not gt_found or not chain_found:
            skipped.append((stem, "no GT" if not gt_found else "no chain pred"))
            rows.append({
                "image": stem,
                "gt_body_px": 0,
                "chain_pred_px": 0,
                "intersection_px": 0,
                "chain_coverage_pct": 0.0,
                "chain_pred_precision_pct": 0.0,
                "gt_mask_found": gt_found,
                "chain_pred_found": chain_found,
            })
            continue

        image_np = np.array(Image.open(img_path).convert("RGB"))
        gt_color = np.array(Image.open(gt_path).convert("RGB"))

        # Resize GT mask to image size if needed
        if gt_color.shape[:2] != image_np.shape[:2]:
            gt_color = np.array(
                Image.fromarray(gt_color).resize(
                    (image_np.shape[1], image_np.shape[0]), Image.NEAREST
                )
            )

        body_mask = extract_body_mask(gt_color, body_color)

        chain_pred_img = np.array(Image.open(chain_path).convert("L"))
        # Resize chain pred to image size if needed
        if chain_pred_img.shape[:2] != image_np.shape[:2]:
            chain_pred_img = np.array(
                Image.fromarray(chain_pred_img).resize(
                    (image_np.shape[1], image_np.shape[0]), Image.NEAREST
                )
            )
        chain_pred = chain_pred_img > 0

        pseudogt = body_mask & chain_pred

        gt_body_px = int(body_mask.sum())
        chain_pred_px = int(chain_pred.sum())
        intersection_px = int(pseudogt.sum())

        chain_coverage_pct = 100.0 * intersection_px / gt_body_px if gt_body_px > 0 else 0.0
        chain_pred_precision_pct = 100.0 * intersection_px / chain_pred_px if chain_pred_px > 0 else 0.0

        row = {
            "image": stem,
            "gt_body_px": gt_body_px,
            "chain_pred_px": chain_pred_px,
            "intersection_px": intersection_px,
            "chain_coverage_pct": round(chain_coverage_pct, 2),
            "chain_pred_precision_pct": round(chain_pred_precision_pct, 2),
            "gt_mask_found": True,
            "chain_pred_found": True,
        }
        rows.append(row)

        # Save pseudo-GT mask
        pseudogt_img = Image.fromarray((pseudogt.astype(np.uint8)) * 255)
        pseudogt_img.save(os.path.join(output_dir, f"{stem}_pseudogt_chain.png"))

        # Save visualization
        fig = make_visualization(image_np, body_mask, chain_pred, pseudogt, stem, row)
        fig.savefig(os.path.join(vis_dir, f"{stem}.jpg"), dpi=100, bbox_inches="tight")
        plt.close(fig)

    # Write CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    valid = [r for r in rows if r["gt_mask_found"] and r["chain_pred_found"]]
    print(f"\n{'=' * 60}")
    print(f"Processed : {len(valid)} / {len(image_files)} images")
    if skipped:
        print(f"Skipped   : {len(skipped)}  (missing GT or chain pred)")
        for name, reason in skipped[:5]:
            print(f"  {name}: {reason}")
        if len(skipped) > 5:
            print(f"  ... and {len(skipped) - 5} more")
    if valid:
        coverages = [r["chain_coverage_pct"] for r in valid]
        precisions = [r["chain_pred_precision_pct"] for r in valid]
        has_chain = sum(1 for r in valid if r["intersection_px"] > 0)
        print(f"\nChain detections (intersection > 0): {has_chain} / {len(valid)} images")
        print(f"Chain coverage   (intersection / GT body)  — mean: {np.mean(coverages):.1f}%  median: {np.median(coverages):.1f}%  max: {max(coverages):.1f}%")
        print(f"Chain precision  (intersection / chain pred) — mean: {np.mean(precisions):.1f}%  median: {np.median(precisions):.1f}%  min: {min(precisions):.1f}%")
        print(f"\nHigh precision (>80%) means chain predictions land well within the GT body.")
        print(f"High coverage (>20%) means chains make up a significant portion of the body — worth tracking.")
    print(f"\nOutputs saved to: {output_dir}")
    print(f"  CSV     : {csv_path}")
    print(f"  Vis     : {vis_dir}/")
    print(f"{'=' * 60}")


def _find_file(directory: Path, stem: str) -> Path | None:
    """Find a file in directory matching stem with any image extension."""
    for ext in IMAGE_EXTENSIONS | {".PNG", ".JPG", ".JPEG"}:
        p = directory / (stem + ext)
        if p.exists():
            return p
    return None


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Intersect SAM3 'chain' predictions with GT body masks to produce "
            "candidate pseudo-GT chain masks, and analyse whether they look sensible."
        )
    )
    parser.add_argument(
        "--image-dir", required=True,
        help="Folder of input images (used for visualisation).",
    )
    parser.add_argument(
        "--gt-mask-dir", required=True,
        help="Folder of GT colour annotation PNGs (e.g. red=body, black=background).",
    )
    parser.add_argument(
        "--chain-pred-dir", required=True,
        help="Folder of binary chain prediction PNGs from run_inference.py --prompt chain.",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Where to write visualisations, pseudo-GT masks, and the CSV.",
    )
    parser.add_argument(
        "--body-color", default="255,0,0",
        help="RGB of the body region in GT colour masks (default: 255,0,0 for red).",
    )
    args = parser.parse_args()

    body_color = tuple(int(x) for x in args.body_color.split(","))
    assert len(body_color) == 3, "--body-color must be three comma-separated integers, e.g. 255,0,0"

    run_analysis(
        image_dir=args.image_dir,
        gt_mask_dir=args.gt_mask_dir,
        chain_pred_dir=args.chain_pred_dir,
        output_dir=args.output_dir,
        body_color=body_color,
    )


if __name__ == "__main__":
    main()
