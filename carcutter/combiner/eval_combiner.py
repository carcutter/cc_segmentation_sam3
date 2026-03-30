#!/usr/bin/env python3
"""
Combiner evaluation script.

Runs the trained MaskCombiner on the val set and compares three outputs:
  1. Fine-tuned model alone (M_finetuned)
  2. Union of all 15 base-model sub-component prompts (M_union)
  3. Combiner output (M_combined)

Outputs:
  comparison.csv        — per-image metrics for all three
  summary.txt           — aggregate stats table
  examples_improved.jpg — images where combiner beats finetuned by most
  examples_degraded.jpg — images where combiner is worse than finetuned

Usage:
    python carcutter/combiner/eval_combiner.py \
        --checkpoint     /path/to/combiner_output/combiner_best.pt \
        --finetuned-dir  /path/to/finetuned_preds \
        --subcomp-dir    /path/to/multi_prompt_analysis/preds \
        --gt-dir         /path/to/masks_foreground_crop \
        --image-dir      /path/to/raw_foreground_crop \
        --val-json       /path/to/sam3_format/annotations/instances_val.json \
        --output-dir     /path/to/combiner_output/eval
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
import torch
from PIL import Image
from tqdm import tqdm

from carcutter.combiner.train_combiner import MaskCombiner, load_binary_mask, _find_file, IMAGE_EXTENSIONS


def mask_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    pred = pred.astype(bool)
    gt   = gt.astype(bool)
    tp = (pred & gt).sum()
    fp = (pred & ~gt).sum()
    fn = (~pred & gt).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    iou       = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4),
            "iou": round(iou, 4), "f1": round(f1, 4)}


def save_comparison_sheet(rows, image_dir, gt_dir, finetuned_dir, subcomp_dir, slugs,
                           combiner_preds, out_path, n=8):
    rows = rows[:n]
    fig, axes = plt.subplots(len(rows), 4, figsize=(12, len(rows) * 3))
    if len(rows) == 1:
        axes = axes[np.newaxis, :]

    for i, row in enumerate(rows):
        stem = row["image"]
        img_path = _find_file(image_dir, stem)
        img = np.array(Image.open(img_path).convert("RGB")) if img_path else np.zeros((100,100,3), dtype=np.uint8)
        h, w = img.shape[:2]

        gt_path = _find_file(gt_dir, stem)
        gt = load_binary_mask(gt_path) if gt_path else np.zeros((h, w), dtype=bool)
        if gt.shape != (h, w):
            gt = np.array(Image.fromarray(gt.astype(np.uint8)*255).resize((w,h), Image.NEAREST)) > 0

        ft_path = _find_file(finetuned_dir, stem)
        ft = load_binary_mask(ft_path) if ft_path else np.zeros((h, w), dtype=bool)
        if ft.shape != (h, w):
            ft = np.array(Image.fromarray(ft.astype(np.uint8)*255).resize((w,h), Image.NEAREST)) > 0

        combined = combiner_preds.get(stem, np.zeros((h, w), dtype=bool))
        if combined.shape != (h, w):
            combined = np.array(Image.fromarray(combined.astype(np.uint8)*255).resize((w,h), Image.NEAREST)) > 0

        for j, (mask, color, label) in enumerate([
            (gt,       (0.9, 0.2, 0.2), f"GT"),
            (ft,       (0.2, 0.6, 1.0), f"finetuned IoU={row['ft_iou']}"),
            (combined, (0.2, 0.8, 0.2), f"combined IoU={row['comb_iou']}"),
            (combined ^ ft, (1.0, 0.6, 0.0), f"diff (combined⊕finetuned)"),
        ]):
            axes[i, j].imshow(img)
            if mask is not None and mask.any():
                ov = np.zeros((*mask.shape, 4), dtype=np.float32)
                ov[mask] = (*color, 0.5)
                axes[i, j].imshow(ov)
            axes[i, j].set_title(label, fontsize=7)
            axes[i, j].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=90, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",    required=True)
    parser.add_argument("--finetuned-dir", required=True)
    parser.add_argument("--subcomp-dir",   required=True)
    parser.add_argument("--gt-dir",        required=True)
    parser.add_argument("--image-dir",     required=True)
    parser.add_argument("--val-json",      required=True)
    parser.add_argument("--output-dir",    required=True)
    parser.add_argument("--img-size",      type=int, default=512)
    parser.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # Load manifest
    manifest_path = Path(args.subcomp_dir) / "_manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    slugs = list(manifest.keys())

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = MaskCombiner(n_in=1 + len(slugs)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint (epoch {ckpt.get('epoch', '?')}, val IoU={ckpt.get('val_iou', '?'):.4f})")

    # Load val stems
    val_data = json.load(open(args.val_json))
    stems = [Path(img["file_name"]).stem for img in val_data["images"]]
    print(f"Evaluating {len(stems)} val images...")

    finetuned_dir = Path(args.finetuned_dir)
    subcomp_dir   = Path(args.subcomp_dir)
    gt_dir        = Path(args.gt_dir)
    image_dir     = Path(args.image_dir)
    target_size   = (args.img_size, args.img_size)

    rows = []
    combiner_preds = {}  # stem → full-res binary mask

    with torch.no_grad():
        for stem in tqdm(stems, desc="Evaluating"):
            # Load GT (full res for metrics)
            gt_path = _find_file(gt_dir, stem)
            if gt_path is None:
                continue
            gt_full = load_binary_mask(gt_path)

            # Fine-tuned prediction (full res); resize to match GT if shapes differ
            ft_path = _find_file(finetuned_dir, stem)
            if ft_path:
                ft_full = load_binary_mask(ft_path)
                if ft_full.shape != gt_full.shape:
                    ft_full = np.array(Image.fromarray(ft_full.astype(np.uint8) * 255).resize(
                        (gt_full.shape[1], gt_full.shape[0]), Image.NEAREST)) > 0
            else:
                ft_full = np.zeros_like(gt_full)

            # Sub-component union (full res, for reporting)
            union_full = np.zeros_like(gt_full)
            for slug in slugs:
                p = _find_file(subcomp_dir / slug, stem)
                if p:
                    m = load_binary_mask(p)
                    if m.shape != gt_full.shape:
                        m = np.array(Image.fromarray(m.astype(np.uint8)*255).resize(
                            (gt_full.shape[1], gt_full.shape[0]), Image.NEAREST)) > 0
                    union_full |= m

            # Run combiner at training resolution
            ft_r  = load_binary_mask(ft_path, target_size).astype(np.float32) if ft_path else np.zeros(target_size, dtype=np.float32)
            subs  = []
            for slug in slugs:
                p = _find_file(subcomp_dir / slug, stem)
                m = load_binary_mask(p, target_size).astype(np.float32) if p else np.zeros(target_size, dtype=np.float32)
                subs.append(m)
            x = torch.from_numpy(np.stack([ft_r] + subs)).unsqueeze(0).to(device)
            logit = model(x)[0, 0].cpu().numpy()
            pred_small = (1 / (1 + np.exp(-logit))) > 0.5

            # Resize back to full res
            pred_full = np.array(
                Image.fromarray(pred_small.astype(np.uint8) * 255).resize(
                    (gt_full.shape[1], gt_full.shape[0]), Image.NEAREST
                )
            ) > 0
            combiner_preds[stem] = pred_full

            ft_m   = mask_metrics(ft_full,   gt_full)
            union_m = mask_metrics(union_full, gt_full)
            comb_m = mask_metrics(pred_full,  gt_full)

            rows.append({
                "image":          stem,
                "ft_iou":         ft_m["iou"],
                "ft_recall":      ft_m["recall"],
                "ft_precision":   ft_m["precision"],
                "ft_f1":          ft_m["f1"],
                "union_iou":      union_m["iou"],
                "union_recall":   union_m["recall"],
                "union_precision":union_m["precision"],
                "union_f1":       union_m["f1"],
                "comb_iou":       comb_m["iou"],
                "comb_recall":    comb_m["recall"],
                "comb_precision": comb_m["precision"],
                "comb_f1":        comb_m["f1"],
                "iou_delta":      round(comb_m["iou"] - ft_m["iou"], 4),
            })

    # Write CSV
    csv_path = os.path.join(args.output_dir, "comparison.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    def mean(key):
        return round(np.mean([r[key] for r in rows]), 4)

    summary = f"""Mask Combiner Evaluation — Val Set ({len(rows)} images)
{'='*60}

{'Method':<20} {'IoU':>8} {'Recall':>8} {'Precision':>10} {'F1':>8}
{'-'*60}
{'Fine-tuned only':<20} {mean('ft_iou'):>8.3f} {mean('ft_recall'):>8.3f} {mean('ft_precision'):>10.3f} {mean('ft_f1'):>8.3f}
{'Sub-comp union':<20} {mean('union_iou'):>8.3f} {mean('union_recall'):>8.3f} {mean('union_precision'):>10.3f} {mean('union_f1'):>8.3f}
{'Combiner':<20} {mean('comb_iou'):>8.3f} {mean('comb_recall'):>8.3f} {mean('comb_precision'):>10.3f} {mean('comb_f1'):>8.3f}

IoU delta (combiner vs finetuned):
  mean   = {mean('iou_delta'):+.4f}
  improved (delta>0) : {sum(1 for r in rows if r['iou_delta']>0)}/{len(rows)} images
  degraded (delta<0) : {sum(1 for r in rows if r['iou_delta']<0)}/{len(rows)} images
  neutral             : {sum(1 for r in rows if r['iou_delta']==0)}/{len(rows)} images
"""
    print(summary)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as f:
        f.write(summary)

    # Contact sheets
    sorted_by_delta = sorted(rows, key=lambda r: -r["iou_delta"])
    save_comparison_sheet(
        sorted_by_delta[:8], image_dir, gt_dir, finetuned_dir, subcomp_dir, slugs,
        combiner_preds, os.path.join(args.output_dir, "examples_improved.jpg"),
    )
    save_comparison_sheet(
        sorted_by_delta[-8:], image_dir, gt_dir, finetuned_dir, subcomp_dir, slugs,
        combiner_preds, os.path.join(args.output_dir, "examples_degraded.jpg"),
    )


if __name__ == "__main__":
    main()
