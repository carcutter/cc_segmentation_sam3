#!/usr/bin/env python3
"""
Mask combiner training script.

Trains a shallow conv net that takes a stack of binary segmentation masks
(fine-tuned trailer prediction + 15 sub-component base model predictions) as
input and outputs a refined binary trailer mask.

The combiner learns to:
  - Add back missed chain/cable/hitch detail that the fine-tuned model skips
  - Remove over-segmentation regions where the fine-tuned model bleeds outside
    the trailer

Inputs  : [finetuned_pred, chain, safety_chain, cable, wheel, tire, axle,
           hitch, coupler, trailer_tongue, ramp, gate, trailer_frame,
           trailer_floor, jack, fender]  — 16 binary channels, float32
Output  : refined binary mask — 1 channel, sigmoid output
GT      : binary GT body masks from masks_foreground_crop

Usage:
    python carcutter/combiner/train_combiner.py \
        --finetuned-dir  /path/to/finetuned_preds \
        --subcomp-dir    /path/to/multi_prompt_analysis/preds \
        --gt-dir         /path/to/masks_foreground_crop \
        --train-json     /path/to/sam3_format/annotations/instances_train.json \
        --val-json       /path/to/sam3_format/annotations/instances_val.json \
        --output-dir     /path/to/combiner_output \
        [--epochs 30] [--lr 1e-3] [--batch-size 8] [--img-size 512]
"""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MaskCombiner(nn.Module):
    """
    Shallow conv net for mask fusion. Processes local spatial context
    (receptive field ~32px) to handle boundary alignment between input masks.

    16 input channels (1 finetuned + 15 sub-components) → 1 output channel.
    """

    def __init__(self, n_in: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_in, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),  # per-pixel logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    return 1.0 - (2.0 * intersection + eps) / (union + eps)


def combined_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")
    dice = dice_loss(logits, targets).mean()
    return bce + dice


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def _find_file(directory: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTENSIONS | {".PNG", ".JPG", ".JPEG"}:
        p = directory / (stem + ext)
        if p.exists():
            return p
    return None


def load_binary_mask(path: Path, target_size: tuple[int, int] | None = None) -> np.ndarray:
    """Load as boolean, optionally resize."""
    img = Image.open(path).convert("L")
    if target_size is not None:
        img = img.resize((target_size[1], target_size[0]), Image.NEAREST)
    return np.array(img) > 0


class MaskCombinerDataset(Dataset):
    def __init__(
        self,
        stems: list[str],
        finetuned_dir: Path,
        subcomp_dir: Path,
        gt_dir: Path,
        slugs: list[str],
        img_size: int = 512,
    ):
        self.stems = stems
        self.finetuned_dir = finetuned_dir
        self.subcomp_dir = subcomp_dir
        self.gt_dir = gt_dir
        self.slugs = slugs
        self.img_size = img_size
        self.target_size = (img_size, img_size)

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> dict:
        stem = self.stems[idx]

        # Fine-tuned prediction
        ft_path = _find_file(self.finetuned_dir, stem)
        if ft_path is not None:
            ft_mask = load_binary_mask(ft_path, self.target_size).astype(np.float32)
        else:
            ft_mask = np.zeros(self.target_size, dtype=np.float32)

        # Sub-component masks
        sub_masks = []
        for slug in self.slugs:
            p = _find_file(self.subcomp_dir / slug, stem)
            if p is not None:
                m = load_binary_mask(p, self.target_size).astype(np.float32)
            else:
                m = np.zeros(self.target_size, dtype=np.float32)
            sub_masks.append(m)

        # Stack: [16, H, W]
        channels = np.stack([ft_mask] + sub_masks, axis=0)
        x = torch.from_numpy(channels)

        # GT
        gt_path = _find_file(self.gt_dir, stem)
        if gt_path is not None:
            gt = load_binary_mask(gt_path, self.target_size).astype(np.float32)
        else:
            gt = np.zeros(self.target_size, dtype=np.float32)
        y = torch.from_numpy(gt).unsqueeze(0)

        return {"x": x, "y": y, "stem": stem}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5):
    preds = (torch.sigmoid(logits) > threshold).float()
    tp = (preds * targets).sum(dim=(1, 2, 3))
    fp = (preds * (1 - targets)).sum(dim=(1, 2, 3))
    fn = ((1 - preds) * targets).sum(dim=(1, 2, 3))
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    iou = tp / (tp + fp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return {
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "iou": iou.mean().item(),
        "f1": f1.mean().item(),
    }


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def run_epoch(model, loader, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_logits, all_targets = [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, desc="train" if train else "val ", leave=False):
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            logits = model(x)
            loss = combined_loss(logits, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x.size(0)
            all_logits.append(logits.detach().cpu())
            all_targets.append(y.detach().cpu())

    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    metrics = compute_metrics(all_logits, all_targets)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train mask combiner for trailer segmentation refinement.")
    parser.add_argument("--finetuned-dir", required=True, help="Dir of fine-tuned trailer predictions.")
    parser.add_argument("--subcomp-dir", required=True, help="Root dir of multi-prompt base predictions (contains per-slug subdirs + _manifest.json).")
    parser.add_argument("--gt-dir", required=True, help="Dir of binary GT body masks.")
    parser.add_argument("--train-json", required=True, help="COCO instances_train.json with file_name list.")
    parser.add_argument("--val-json", required=True, help="COCO instances_val.json with file_name list.")
    parser.add_argument("--output-dir", required=True, help="Where to write checkpoints and metrics.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load manifest for slug order
    manifest_path = Path(args.subcomp_dir) / "_manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    slugs = list(manifest.keys())
    prompts = [manifest[s]["prompt"] for s in slugs]
    print(f"Sub-component channels ({len(slugs)}): {prompts}")

    # Load train/val stems
    def load_stems(json_path):
        data = json.load(open(json_path))
        return [Path(img["file_name"]).stem for img in data["images"]]

    train_stems = load_stems(args.train_json)
    val_stems   = load_stems(args.val_json)
    print(f"Train: {len(train_stems)}  Val: {len(val_stems)}")

    finetuned_dir = Path(args.finetuned_dir)
    subcomp_dir   = Path(args.subcomp_dir)
    gt_dir        = Path(args.gt_dir)

    train_ds = MaskCombinerDataset(train_stems, finetuned_dir, subcomp_dir, gt_dir, slugs, args.img_size)
    val_ds   = MaskCombinerDataset(val_stems,   finetuned_dir, subcomp_dir, gt_dir, slugs, args.img_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device(args.device)
    model = MaskCombiner(n_in=1 + len(slugs)).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    log_rows = []
    best_val_iou = 0.0

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, train=True)
        val_metrics   = run_epoch(model, val_loader,   optimizer, device, train=False)
        scheduler.step()

        row = {"epoch": epoch}
        row.update({f"train_{k}": round(v, 4) for k, v in train_metrics.items()})
        row.update({f"val_{k}":   round(v, 4) for k, v in val_metrics.items()})
        log_rows.append(row)

        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"loss {train_metrics['loss']:.4f} → {val_metrics['loss']:.4f}  "
            f"IoU {train_metrics['iou']:.3f} → {val_metrics['iou']:.3f}  "
            f"recall {train_metrics['recall']:.3f} → {val_metrics['recall']:.3f}  "
            f"precision {train_metrics['precision']:.3f} → {val_metrics['precision']:.3f}"
        )

        # Save best
        if val_metrics["iou"] > best_val_iou:
            best_val_iou = val_metrics["iou"]
            ckpt_path = os.path.join(args.output_dir, "combiner_best.pt")
            torch.save({"epoch": epoch, "model": model.state_dict(), "val_iou": best_val_iou}, ckpt_path)
            print(f"  ✓ saved best checkpoint (val IoU={best_val_iou:.4f})")

    # Save final checkpoint
    torch.save({"epoch": args.epochs, "model": model.state_dict()},
               os.path.join(args.output_dir, "combiner_final.pt"))

    # Write training log
    log_path = os.path.join(args.output_dir, "training_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"\nDone. Best val IoU: {best_val_iou:.4f}")
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
