# Training Guide Reference

## Config file locations

```
sam3/train/configs/
├── holes/freeze/holes_finetune_with_freezing.yaml
├── outline/freeze/outline_finetune_with_freezing.yaml
└── mirror/mirror_finetune.yaml
```

## Full freeze_config options

```yaml
freeze_config:
  vision_backbone: true         # Freeze ViT image encoder (~600M params)
  language_backbone: true       # Freeze CLIP text encoder (~150M params)
  geometry_encoder: true        # Freeze box/point encoder
  transformer_encoder: false    # Train fusion layers
  transformer_decoder: false    # Train decoder
  segmentation_head: false      # Train mask predictor
  dot_prod_scoring: false       # Train classifier

  # Advanced options
  vision_backbone_num_frozen_layers: null  # Freeze first N ViT blocks only
  custom_patterns: []           # Additional glob patterns to freeze
  freeze_all_except: []         # Adapter-style: freeze all except these
  unfreeze_patterns: []         # Override - always unfreeze these
```

## Freeze strategies

| Strategy | Frozen components | VRAM | Use case |
|---|---|---|---|
| Full fine-tune | none | 24GB+ | Enough data, best quality |
| Backbone frozen | vision + language | 12-16GB | Limited GPU, good starting point |
| Decoder only | vision + language + geometry | 8-12GB | Very limited GPU |
| Adapter | freeze_all_except: [segmentation_head.*] | 6-8GB | Minimal data |
| Gradual | decrease vision_backbone_num_frozen_layers | varies | Curriculum approach |

## Key learning rate settings

```yaml
scratch:
  lr_scale: 0.5                    # Global multiplier
  lr_transformer: 8.0e-5           # Decoder learning rate (* lr_scale)
  lr_vision_backbone: 2.5e-4       # Image encoder (* lr_scale)
  lr_language_backbone: 5.0e-5     # Text encoder (* lr_scale)
  lrd_vision_backbone: 0.95        # Layer-wise decay for vision backbone
  wd: 0.05                         # Weight decay
  scheduler_timescale: 20
  scheduler_warmup: 20
  scheduler_cooldown: 20
```

## Data loading settings

```yaml
data:
  train:
    batch_size: 12       # Reduce if OOM
    num_workers: 10
    shuffle: True
    dataset:
      img_folder: ${paths.dataset_root}/train/images
      ann_file: ${paths.dataset_root}/sam3_format/annotations/instances_train.json
  val:
    batch_size: 1
    num_workers: 0
    dataset:
      img_folder: ${paths.dataset_root}/val/images
      ann_file: ${paths.dataset_root}/sam3_format/annotations/instances_val.json
```

## Training output structure

```
{experiment_log_dir}/
├── config.yaml               # Original config
├── config_resolved.yaml      # Fully resolved config
├── checkpoints/              # Saved model weights
├── tensorboard/              # TensorBoard event files
└── logs/                     # Text logs
```

## Key metrics tracked

- **IoU** — Intersection over Union on full mask
- **Contour IoU** — IoU on eroded mask boundary (kernel=5, iterations=2); indicates edge quality
- **COCO mAP** — bbox detection mAP (validation)

## Trainer classes

- `sam3.train.trainer.Trainer` — standard trainer
- `sam3.train.trainer_with_freezing.TrainerWithFreezing` — use this when freeze_config is set

Set in config:
```yaml
trainer:
  _target_: sam3.train.trainer_with_freezing.TrainerWithFreezing
  max_epochs: 600
```

## SLURM cluster launch

```bash
python sam3/train/train.py \
  -c sam3/train/configs/{task}/freeze/{task}_finetune_with_freezing.yaml \
  --use-cluster 1 \
  --partition gpu \
  --num-gpus 8 \
  --num-nodes 2
```
