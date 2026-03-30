#!/usr/bin/env python3
"""
Multi-prompt union analysis script.

Unions all per-prompt masks produced by run_multi_prompt_inference.py and measures
recall/precision/IoU vs the GT trailer body mask. Also runs a greedy ablation to
identify which prompts contribute most to recall.

Workflow:
  1. Run run_multi_prompt_inference.py to produce per-prompt mask subfolders.
  2. Run this script to compute union metrics and prompt contribution analysis.

Two-pass algorithm:
  - Pass 1: per-image union metrics + per-prompt raw GT coverage.
  - Pass 2: cumulative union recall/precision as prompts are added in coverage-ranked order
            (greedy ablation).

Usage:
    python carcutter/eval/analyze_prompt_union.py \\
        --image-dir     /path/to/raw_foreground_crop \\
        --gt-mask-dir   /path/to/masks_foreground_crop \\
        --pred-dir      /path/to/multi_prompt_preds \\
        --output-dir    /path/to/analysis_output \\
        [--gt-body-color "255,255,255"]

Outputs:
    metrics_per_image.csv       — precision/recall/IoU/F1 per image (union of all prompts)
    prompt_contribution.csv     — per-prompt raw coverage + median marginal recall in greedy order
    greedy_ablation.csv         — cumulative recall/precision as prompts added greedily
    summary.txt                 — aggregate stats
    visualizations/{stem}.jpg   — original | GT | union | per-prompt grid
    examples_best_recall.jpg    — contact sheet of 5 highest-recall images
    examples_worst_recall.jpg   — contact sheet of 5 lowest-recall images (GT area > 0)
    greedy_ablation_plot.png    — recall vs #prompts and precision vs recall curve
"""

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ---------------------------------------------------------------------------
# Helpers (reused pattern from analyze_chain_pseudogt.py)
# ---------------------------------------------------------------------------

def load_binary(path: Path) -> np.ndarray:
    """Load any image as a boolean mask (True where non-zero)."""
    return np.array(Image.open(path).convert("L")) > 0


def extract_body_mask(gt_color: np.ndarray, body_rgb: tuple[int, int, int]) -> np.ndarray:
    r, g, b = body_rgb
    return (
        (gt_color[:, :, 0] == r) &
        (gt_color[:, :, 1] == g) &
        (gt_color[:, :, 2] == b)
    )


def _find_file(directory: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTENSIONS | {".PNG", ".JPG", ".JPEG"}:
        p = directory / (stem + ext)
        if p.exists():
            return p
    return None


def load_gt_mask(
    gt_mask_dir: Path,
    stem: str,
    body_rgb: tuple[int, int, int],
    target_shape: tuple[int, int] | None = None,
) -> np.ndarray | None:
    """Load GT mask as boolean array. Returns None if file not found."""
    gt_path = _find_file(gt_mask_dir, stem)
    if gt_path is None:
        return None
    gt_color = np.array(Image.open(gt_path).convert("RGB"))
    if target_shape is not None and gt_color.shape[:2] != target_shape:
        gt_color = np.array(
            Image.fromarray(gt_color).resize(
                (target_shape[1], target_shape[0]), Image.NEAREST
            )
        )
    return extract_body_mask(gt_color, body_rgb)


def load_pred_mask(pred_dir: Path, slug: str, stem: str) -> np.ndarray:
    """Load a per-prompt prediction mask. Returns all-False if file not found."""
    pred_path = _find_file(pred_dir / slug, stem)
    if pred_path is None:
        return None
    return load_binary(pred_path)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

PROMPT_COLORS = [
    (0.9, 0.2, 0.2),
    (0.2, 0.8, 0.2),
    (0.2, 0.5, 1.0),
    (1.0, 0.7, 0.0),
    (0.8, 0.2, 0.8),
    (0.0, 0.8, 0.8),
    (1.0, 0.4, 0.0),
    (0.5, 1.0, 0.0),
    (0.6, 0.0, 1.0),
    (1.0, 0.0, 0.5),
    (0.0, 0.5, 0.0),
    (0.5, 0.3, 0.0),
    (0.0, 0.3, 0.7),
    (0.7, 0.7, 0.2),
    (0.4, 0.4, 0.4),
]


def _overlay(ax, image: np.ndarray, mask: np.ndarray | None, color: tuple, alpha: float = 0.55):
    ax.imshow(image)
    if mask is not None and mask.any():
        overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
        overlay[mask] = (*color, alpha)
        ax.imshow(overlay)
    ax.axis("off")


def save_visualization(
    image_np: np.ndarray,
    gt_mask: np.ndarray,
    union_mask: np.ndarray,
    per_prompt_masks: dict[str, np.ndarray],
    prompts: list[str],
    title: str,
    out_path: str,
) -> None:
    n_prompts = len(prompts)
    # Layout: row 0 → original | GT | union (+ empty cells to fill row)
    #         rows 1+ → per-prompt grid (5 per row)
    COLS = 5
    prompt_rows = (n_prompts + COLS - 1) // COLS
    total_rows = 1 + prompt_rows

    fig, axes = plt.subplots(total_rows, COLS, figsize=(COLS * 3, total_rows * 3))
    if total_rows == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(title, fontsize=8)

    # Row 0: original, GT, union, blanks
    _overlay(axes[0, 0], image_np, None, (0, 0, 0))
    axes[0, 0].set_title("Original", fontsize=7)

    _overlay(axes[0, 1], image_np, gt_mask, (0.9, 0.2, 0.2))
    axes[0, 1].set_title("GT body", fontsize=7)

    _overlay(axes[0, 2], image_np, union_mask, (0.2, 0.7, 0.2))
    axes[0, 2].set_title("Union (all prompts)", fontsize=7)

    for col in range(3, COLS):
        axes[0, col].axis("off")

    # Remaining rows: per-prompt
    for idx, prompt in enumerate(prompts):
        row = 1 + idx // COLS
        col = idx % COLS
        mask = per_prompt_masks.get(prompt)
        color = PROMPT_COLORS[idx % len(PROMPT_COLORS)]
        _overlay(axes[row, col], image_np, mask, color)
        axes[row, col].set_title(prompt, fontsize=6)

    # Hide any unused cells in last prompt row
    last_used = n_prompts % COLS
    if last_used != 0:
        for col in range(1 + last_used, COLS):
            axes[-1, col].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=80, bbox_inches="tight")
    plt.close(fig)


def save_contact_sheet(
    rows: list[dict],
    image_dir: Path,
    gt_mask_dir: Path,
    pred_dir: Path,
    slugs: dict[str, str],
    body_rgb: tuple[int, int, int],
    out_path: str,
    n: int = 5,
) -> None:
    """Save a contact sheet of n images showing original | GT | union."""
    rows = rows[:n]
    fig, axes = plt.subplots(len(rows), 3, figsize=(9, len(rows) * 3))
    if len(rows) == 1:
        axes = axes[np.newaxis, :]

    for i, row in enumerate(rows):
        stem = row["image"]
        img_path = _find_file(image_dir, stem)
        if img_path is None:
            for ax in axes[i]:
                ax.axis("off")
            continue

        image_np = np.array(Image.open(img_path).convert("RGB"))
        gt_mask = load_gt_mask(gt_mask_dir, stem, body_rgb, image_np.shape[:2])

        union = np.zeros(image_np.shape[:2], dtype=bool)
        for slug in slugs.values():
            pred = load_pred_mask(pred_dir, slug, stem)
            if pred is not None:
                union |= pred

        recall = row.get("recall", 0)
        _overlay(axes[i, 0], image_np, None, (0, 0, 0))
        axes[i, 0].set_title(f"{stem}  recall={recall:.2f}", fontsize=7)
        _overlay(axes[i, 1], image_np, gt_mask, (0.9, 0.2, 0.2))
        axes[i, 1].set_title("GT", fontsize=7)
        _overlay(axes[i, 2], image_np, union, (0.2, 0.7, 0.2))
        axes[i, 2].set_title("Union", fontsize=7)

    plt.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_analysis(
    image_dir: str,
    gt_mask_dir: str,
    pred_dir: str,
    output_dir: str,
    body_rgb: tuple[int, int, int],
    save_visualizations: bool = True,
) -> None:
    image_dir = Path(image_dir)
    gt_mask_dir = Path(gt_mask_dir)
    pred_dir = Path(pred_dir)
    vis_dir = Path(output_dir) / "visualizations"
    os.makedirs(vis_dir, exist_ok=True)

    # Load manifest for prompt → slug mapping
    manifest_path = pred_dir / "_manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: No _manifest.json found in {pred_dir}. Run run_multi_prompt_inference.py first.")
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Order: preserve original prompt order from manifest (insertion order, Python 3.7+)
    slugs: dict[str, str] = {entry["prompt"]: slug for slug, entry in manifest.items()}
    prompts = list(slugs.keys())

    image_files = sorted(
        p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_files:
        print(f"No images found in {image_dir}")
        return

    print(f"Found {len(image_files)} images, {len(prompts)} prompts")
    print(f"GT body color: {body_rgb}")

    # -----------------------------------------------------------------------
    # Pass 1: per-image metrics + per-prompt GT coverage
    # -----------------------------------------------------------------------
    metrics_rows = []
    # prompt_gt_coverage[p] = list of (pred ∩ gt) / gt_area per image (skip if gt_area==0)
    prompt_gt_coverage: dict[str, list[float]] = {p: [] for p in prompts}

    print("\nPass 1: computing per-image metrics...")
    for img_path in tqdm(image_files, desc="Analysing"):
        stem = img_path.stem
        image_np = np.array(Image.open(img_path).convert("RGB"))
        gt_mask = load_gt_mask(gt_mask_dir, stem, body_rgb, image_np.shape[:2])

        if gt_mask is None:
            metrics_rows.append({
                "image": stem,
                "precision": 0.0, "recall": 0.0, "iou": 0.0, "f1": 0.0,
                "union_area": 0, "gt_area": 0, "gt_found": False,
            })
            continue

        gt_area = int(gt_mask.sum())
        union = np.zeros_like(gt_mask)
        per_prompt_masks: dict[str, np.ndarray] = {}

        for prompt, slug in slugs.items():
            pred = load_pred_mask(pred_dir, slug, stem)
            if pred is None:
                pred = np.zeros_like(gt_mask)
            # Resize pred to image size if needed
            if pred.shape != gt_mask.shape:
                pred = np.array(
                    Image.fromarray(pred.astype(np.uint8) * 255).resize(
                        (gt_mask.shape[1], gt_mask.shape[0]), Image.NEAREST
                    )
                ) > 0
            per_prompt_masks[prompt] = pred
            union |= pred
            coverage = float((pred & gt_mask).sum()) / gt_area if gt_area > 0 else 0.0
            prompt_gt_coverage[prompt].append(coverage)

        union_area = int(union.sum())
        intersection = int((union & gt_mask).sum())
        denom_iou = union_area + gt_area - intersection

        precision = intersection / union_area if union_area > 0 else 0.0
        recall = intersection / gt_area if gt_area > 0 else 0.0
        iou = intersection / denom_iou if denom_iou > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics_rows.append({
            "image": stem,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "iou": round(iou, 4),
            "f1": round(f1, 4),
            "union_area": union_area,
            "gt_area": gt_area,
            "gt_found": True,
        })

        # Visualisation (skip images with no GT area to keep output manageable)
        if save_visualizations and gt_area > 0:
            title = (
                f"{stem}  |  recall={recall:.2f}  precision={precision:.2f}  "
                f"IoU={iou:.2f}  gt={gt_area:,}px  union={union_area:,}px"
            )
            save_visualization(
                image_np, gt_mask, union, per_prompt_masks, prompts,
                title, str(vis_dir / f"{stem}.jpg"),
            )

    # Write metrics_per_image.csv
    metrics_csv = os.path.join(output_dir, "metrics_per_image.csv")
    with open(metrics_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "precision", "recall", "iou", "f1", "union_area", "gt_area", "gt_found"])
        writer.writeheader()
        writer.writerows(metrics_rows)

    # Per-prompt contribution
    prompt_contribution_rows = []
    for prompt in prompts:
        coverages = prompt_gt_coverage[prompt]
        raw_coverage_mean = float(np.mean(coverages)) if coverages else 0.0
        raw_coverage_median = float(np.median(coverages)) if coverages else 0.0
        # Precision within GT = of the pixels in pred, how many are in GT
        # (computed separately in pass 2; approximate here as raw_coverage for now)
        prompt_contribution_rows.append({
            "prompt": prompt,
            "slug": slugs[prompt],
            "mean_gt_coverage_pct": round(raw_coverage_mean * 100, 2),
            "median_gt_coverage_pct": round(raw_coverage_median * 100, 2),
            "images_with_detection_pct": round(
                100.0 * sum(1 for c in coverages if c > 0) / len(coverages)
                if coverages else 0.0, 2
            ),
        })

    # Sort by mean coverage descending (this is the greedy order)
    prompt_contribution_rows.sort(key=lambda r: -r["mean_gt_coverage_pct"])
    greedy_order = [r["prompt"] for r in prompt_contribution_rows]

    # -----------------------------------------------------------------------
    # Pass 2: greedy ablation — single pass per image, accumulate running union
    # O(N×P) file loads instead of O(N×P²)
    # -----------------------------------------------------------------------
    print("\nPass 2: greedy ablation...")
    valid_metrics = [r for r in metrics_rows if r["gt_found"] and r["gt_area"] > 0]

    P = len(greedy_order)
    step_recalls     = [[] for _ in range(P)]
    step_precisions  = [[] for _ in range(P)]
    step_marginals   = [[] for _ in range(P)]

    for row in tqdm(valid_metrics, desc="Greedy ablation"):
        stem = row["image"]
        gt_mask = load_gt_mask(gt_mask_dir, stem, body_rgb)
        if gt_mask is None:
            continue
        gt_area = float(gt_mask.sum())

        union = np.zeros_like(gt_mask)
        prev_recall = 0.0

        for k, prompt in enumerate(greedy_order):
            pred = load_pred_mask(pred_dir, slugs[prompt], stem)
            if pred is not None:
                if pred.shape != gt_mask.shape:
                    pred = np.array(
                        Image.fromarray(pred.astype(np.uint8) * 255).resize(
                            (gt_mask.shape[1], gt_mask.shape[0]), Image.NEAREST
                        )
                    ) > 0
                union |= pred

            union_area = float(union.sum())
            intersection = float((union & gt_mask).sum())
            recall    = intersection / gt_area if gt_area > 0 else 0.0
            precision = intersection / union_area if union_area > 0 else 0.0

            step_recalls[k].append(recall)
            step_precisions[k].append(precision)
            step_marginals[k].append(recall - prev_recall)
            prev_recall = recall

    ablation_rows = []
    for k, prompt in enumerate(greedy_order):
        ablation_rows.append({
            "step": k + 1,
            "prompt": prompt,
            "slug": slugs[prompt],
            "mean_cumulative_recall":    round(float(np.mean(step_recalls[k])), 4),
            "median_cumulative_recall":  round(float(np.median(step_recalls[k])), 4),
            "mean_cumulative_precision": round(float(np.mean(step_precisions[k])), 4),
            "mean_marginal_recall_gain":   round(float(np.mean(step_marginals[k])), 4),
            "median_marginal_recall_gain": round(float(np.median(step_marginals[k])), 4),
        })

    # Backfill exact marginal recall into prompt_contribution_rows
    marginal_by_prompt = {r["prompt"]: r["median_marginal_recall_gain"] for r in ablation_rows}
    for row in prompt_contribution_rows:
        row["median_marginal_recall_gain"] = marginal_by_prompt.get(row["prompt"], 0.0)

    # Write prompt_contribution.csv
    contrib_csv = os.path.join(output_dir, "prompt_contribution.csv")
    fieldnames = ["prompt", "slug", "mean_gt_coverage_pct", "median_gt_coverage_pct",
                  "images_with_detection_pct", "median_marginal_recall_gain"]
    with open(contrib_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(prompt_contribution_rows)

    # Write greedy_ablation.csv
    ablation_csv = os.path.join(output_dir, "greedy_ablation.csv")
    with open(ablation_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ablation_rows[0].keys()))
        writer.writeheader()
        writer.writerows(ablation_rows)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    valid = [r for r in metrics_rows if r["gt_found"] and r["gt_area"] > 0]
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("Multi-Prompt Union Analysis Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total images     : {len(metrics_rows)}\n")
        f.write(f"With GT mask     : {sum(1 for r in metrics_rows if r['gt_found'])}\n")
        f.write(f"With GT area > 0 : {len(valid)}\n\n")
        if valid:
            recalls = [r["recall"] for r in valid]
            precisions = [r["precision"] for r in valid]
            ious = [r["iou"] for r in valid]
            f1s = [r["f1"] for r in valid]
            f.write(f"Union of all {len(prompts)} prompts:\n")
            f.write(f"  Recall    mean={np.mean(recalls):.3f}  median={np.median(recalls):.3f}  min={min(recalls):.3f}  max={max(recalls):.3f}\n")
            f.write(f"  Precision mean={np.mean(precisions):.3f}  median={np.median(precisions):.3f}\n")
            f.write(f"  IoU       mean={np.mean(ious):.3f}  median={np.median(ious):.3f}\n")
            f.write(f"  F1        mean={np.mean(f1s):.3f}  median={np.median(f1s):.3f}\n\n")
            f.write("Greedy prompt order (by coverage):\n")
            for row in ablation_rows:
                f.write(
                    f"  {row['step']:>2}. {row['prompt']:<22} "
                    f"cumulative recall={row['mean_cumulative_recall']:.3f}  "
                    f"marginal gain={row['mean_marginal_recall_gain']:.3f}\n"
                )

    print(open(summary_path).read())

    # -----------------------------------------------------------------------
    # Greedy ablation plot
    # -----------------------------------------------------------------------
    steps = [r["step"] for r in ablation_rows]
    cum_recalls = [r["mean_cumulative_recall"] for r in ablation_rows]
    cum_precisions = [r["mean_cumulative_precision"] for r in ablation_rows]
    prompt_labels = [r["prompt"] for r in ablation_rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(steps, cum_recalls, "o-", color="steelblue", label="Recall")
    ax1.plot(steps, cum_precisions, "s--", color="tomato", label="Precision")
    ax1.set_xticks(steps)
    ax1.set_xticklabels(prompt_labels, rotation=45, ha="right", fontsize=7)
    ax1.set_xlabel("Prompts added (greedy order)")
    ax1.set_ylabel("Mean value across images")
    ax1.set_title("Greedy ablation: recall & precision vs prompts added")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)

    ax2.plot(cum_recalls, cum_precisions, "o-", color="purple")
    for i, label in enumerate(prompt_labels):
        ax2.annotate(label, (cum_recalls[i], cum_precisions[i]),
                     textcoords="offset points", xytext=(4, 4), fontsize=6)
    ax2.set_xlabel("Mean Recall")
    ax2.set_ylabel("Mean Precision")
    ax2.set_title("Recall–Precision trade-off (greedy prompt addition)")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1.05)
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "greedy_ablation_plot.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Best / worst recall contact sheets
    # -----------------------------------------------------------------------
    if save_visualizations:
        sorted_valid = sorted(valid, key=lambda r: -r["recall"])
        save_contact_sheet(
            sorted_valid[:5], image_dir, gt_mask_dir, pred_dir, slugs, body_rgb,
            os.path.join(output_dir, "examples_best_recall.jpg"),
        )
        worst = [r for r in sorted_valid if r["gt_area"] > 0]
        save_contact_sheet(
            worst[-5:], image_dir, gt_mask_dir, pred_dir, slugs, body_rgb,
            os.path.join(output_dir, "examples_worst_recall.jpg"),
        )

    print(f"\nOutputs saved to: {output_dir}")
    print(f"  metrics_per_image.csv    : {metrics_csv}")
    print(f"  prompt_contribution.csv  : {contrib_csv}")
    print(f"  greedy_ablation.csv      : {ablation_csv}")
    print(f"  greedy_ablation_plot.png : {plot_path}")
    print(f"  visualizations/          : {vis_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Union all per-prompt masks from run_multi_prompt_inference.py, "
            "measure recall/precision vs GT, and run greedy ablation."
        )
    )
    parser.add_argument("--image-dir", required=True, help="Folder of original images.")
    parser.add_argument("--gt-mask-dir", required=True, help="Folder of GT body masks.")
    parser.add_argument(
        "--pred-dir", required=True,
        help="Root output folder from run_multi_prompt_inference.py (contains per-prompt subfolders).",
    )
    parser.add_argument("--output-dir", required=True, help="Where to write analysis outputs.")
    parser.add_argument(
        "--gt-body-color", default="255,255,255",
        help="RGB of the body region in GT masks (default: 255,255,255 for white/binary).",
    )
    parser.add_argument(
        "--no-vis", action="store_true",
        help="Skip per-image visualizations and contact sheets (much faster).",
    )
    args = parser.parse_args()

    body_rgb = tuple(int(x) for x in args.gt_body_color.split(","))
    assert len(body_rgb) == 3, "--gt-body-color must be three comma-separated integers"

    run_analysis(
        image_dir=args.image_dir,
        gt_mask_dir=args.gt_mask_dir,
        pred_dir=args.pred_dir,
        output_dir=args.output_dir,
        body_rgb=body_rgb,
        save_visualizations=not args.no_vis,
    )


if __name__ == "__main__":
    main()
