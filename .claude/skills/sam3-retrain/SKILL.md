---
name: sam3-retrain
description: Guides end-to-end retraining of SAM3 segmentation models for car parts (holes, outline, mirror, trailer). Use when asked to retrain, fine-tune, or prepare training data for segmentation, or when asked to "run the training pipeline", "prepare SAM3 data", "evaluate a segmentation model", or "set up a training config". Covers data preparation, COCO annotation conversion, training with optional parameter freezing, and evaluation. For label quality analysis and annotation review, use the sam3-label-review skill instead.
metadata:
  author: carcutter
  version: 1.0.0
  category: ml-training
  tags: [segmentation, sam3, training, data-preparation]
---

# SAM3 Segmentation Retraining

## CRITICAL: Ask first

Before taking any action, confirm:
1. Which task? (`holes`, `outline`, `mirror`, or `trailer`)
2. Is COCO data already prepared, or does data prep need to run?
3. How many GPUs and approximate VRAM per GPU?
4. Does a training config already exist, or does one need updating?

---

## Phase 1: Data Preparation

Only needed when raw data has changed. All scripts run with `conda run -n sam3`. See `references/data-pipeline.md` for full details.

### Step 0 — Extract binary masks (skip if masks are already binary)
```bash
conda run -n sam3 python carcutter/data_preparation/extract_binary_mask.py \
  --input-dir data/car_segmentation/<batch>/masks \
  --output-dir data/car_segmentation/<batch>/binary_masks \
  --mode all_except_background
```

### Step 1 — Foreground crop (skip if not cropping to object bbox)
```bash
conda run -n sam3 python carcutter/data_preparation/crop_to_outline_bbox.py \
  --raw-dir data/car_segmentation/<batch>/raw_images \
  --mask-dir data/car_segmentation/<batch>/binary_masks \
  --output-dir data/car_segmentation/<batch>
```
Output: `raw_foreground_crop/` and `masks_foreground_crop/` in the batch folder.

### Step 2 — Train/val split
```bash
conda run -n sam3 python carcutter/data_preparation/prepare_splits/create_train_val_split_general.py \
  --raw-dir data/car_segmentation/<batch>/raw_foreground_crop \
  --mask-dir data/car_segmentation/<batch>/masks_foreground_crop \
  --output-dir data/training/<task>/<task>_<date> \
  --task-name <task>
```
Output: `data/training/{task}/{task}_{date}/train/` and `val/` with `images/` + `labels/`.

### Step 3 — Convert to COCO/RLE format
```bash
conda run -n sam3 python carcutter/data_preparation/prepare_sam3_format/prepare_sam3_dataset.py \
  --data-root data/training/<task>/<task>_<date> \
  --task-name <task> --category-names <task> --min-area 100 --min-hole-area 50
```
Output: `data/training/{task}/{task}_{date}/sam3_format/annotations/instances_{train,val}.json`

### Step 4 — Verify (recommended)
```bash
conda run -n sam3 python carcutter/data_preparation/prepare_sam3_format/verify_annotations.py \
  --ann-file data/training/<task>/<task>_<date>/sam3_format/annotations/instances_train.json
conda run -n sam3 python carcutter/data_preparation/prepare_sam3_format/visualize_coco_annotations.py \
  --ann-file data/training/<task>/<task>_<date>/sam3_format/annotations/instances_train.json \
  --img-dir data/training/<task>/<task>_<date>/train/images
```

---

## Phase 2: Training Config

Config location: `sam3/train/configs/{task}/freeze/{task}_finetune_with_freezing.yaml`

Available configs:
- `sam3/train/configs/holes/freeze/holes_finetune_with_freezing.yaml`
- `sam3/train/configs/outline/freeze/outline_finetune_with_freezing.yaml`
- `sam3/train/configs/mirror/freeze/mirror_finetune_with_freezing.yaml`
- `sam3/train/configs/trailer/freeze/trailer_finetune_with_freezing.yaml`

CRITICAL: Always update these fields before training:
- `paths.dataset_root` — path to the dated data folder (e.g. `data/training/holes/260201`)
- `paths.experiment_log_dir` — where checkpoints and logs go
- `data.train.batch_size` — reduce if OOM (default: 12)

Choose freeze strategy based on available VRAM:

| VRAM | Strategy | Config |
|---|---|---|
| 24GB+ | Full fine-tune | all freeze flags false |
| 12-16GB | Backbone frozen | vision_backbone + language_backbone: true |
| 8-12GB | Decoder only | + geometry_encoder: true |

See `references/training-guide.md` for all freeze options and learning rate tuning.

---

## Phase 3: Launch Training

```bash
# Single GPU
python sam3/train/train.py \
  -c sam3/train/configs/{task}/freeze/{task}_finetune_with_freezing.yaml \
  --use-cluster 0 --num-gpus 1

# Multi-GPU
python sam3/train/train.py \
  -c sam3/train/configs/{task}/freeze/{task}_finetune_with_freezing.yaml \
  --use-cluster 0 --num-gpus 4
```

Monitor with TensorBoard: `tensorboard --logdir {experiment_log_dir}`

---

## Phase 4: Evaluation

During training: IoU and contour IoU are logged automatically to TensorBoard.

Post-training: open `evaluate_{task}.ipynb` in Jupyter for per-sample analysis.

---

## Common Issues

**OOM during training**
- Reduce `data.train.batch_size` in the config
- Increase frozen components (`vision_backbone: true`, then `geometry_encoder: true`)

**Missing BPE vocab file**
- Path: `sam3/assets/bpe_simple_vocab_16e6.txt.gz`
- Check README.md for the download command

**RLE annotation errors**
- Run `verify_annotations.py` to diagnose
- Check that `prepare_sam3_dataset.py` used the correct mask color IDs
