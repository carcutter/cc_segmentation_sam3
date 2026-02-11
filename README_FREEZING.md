# SAM3 Parameter Freezing for Fine-tuning

This implementation addresses [GitHub Issue #284](https://github.com/facebookresearch/sam3/issues/284) by providing a systematic way to freeze different parts of the SAM3 network during fine-tuning via YAML configuration.

## Files

- **`trainer_with_freezing.py`** - Modified trainer with parameter freezing support
- **`mirror_finetune_with_freezing.yaml`** - Example config with freezing options

## Installation

1. Copy `trainer_with_freezing.py` to your SAM3 installation:
   ```bash
   cp trainer_with_freezing.py /path/to/sam3/train/trainer_with_freezing.py
   ```

2. Update your config to use the new trainer:
   ```yaml
   trainer:
     _target_: sam3.train.trainer_with_freezing.TrainerWithFreezing
   ```

## Configuration Options

### Main Components

| Option | Description | Parameters (~M) |
|--------|-------------|-----------------|
| `vision_backbone` | ViT image encoder | ~300M |
| `language_backbone` | CLIP text encoder | ~60M |
| `memory_encoder` | Memory encoding components | ~10M |
| `memory_attention` | Memory attention layers | ~5M |
| `mask_decoder` | Mask prediction decoder | ~20M |
| `prompt_encoder` | Prompt encoding layers | ~5M |
| `transformer_decoder` | Transformer decoder layers | ~30M |
| `transformer_encoder` | Transformer encoder layers | ~30M |
| `obj_ptr` | Object pointer components | ~5M |

### Advanced Options

| Option | Description |
|--------|-------------|
| `vision_backbone_num_frozen_layers` | Freeze first N ViT layers (for gradual unfreezing) |
| `custom_patterns` | List of Unix glob patterns to freeze |
| `freeze_all_except` | Freeze everything EXCEPT these patterns (adapter-style) |
| `unfreeze_patterns` | Explicitly unfreeze these patterns (highest priority) |

## Training Strategies

### Strategy 1: Full Fine-tuning
Train all parameters (default behavior):
```yaml
freeze_config:
  vision_backbone: false
  language_backbone: false
  memory_encoder: false
  mask_decoder: false
```
**Best for:** Large datasets, maximum adaptation  
**GPU Memory:** ~24GB+

### Strategy 2: Freeze Vision Backbone
Freeze the image encoder, train decoder components:
```yaml
freeze_config:
  vision_backbone: true
  language_backbone: true
  memory_encoder: false
  mask_decoder: false
```
**Best for:** Small-medium datasets, faster training, less overfitting  
**GPU Memory:** ~12-16GB

### Strategy 3: Decoder-Only Training
Freeze both backbones, train only detection/segmentation heads:
```yaml
freeze_config:
  vision_backbone: true
  language_backbone: true
  memory_encoder: true
  mask_decoder: false
  prompt_encoder: false
```
**Best for:** Very small datasets, rapid prototyping  
**GPU Memory:** ~8-12GB

### Strategy 4: Gradual Unfreezing
Start with many frozen layers, unfreeze progressively:
```yaml
# Epoch 1-100: Freeze first 20 ViT blocks
freeze_config:
  vision_backbone_num_frozen_layers: 20

# Epoch 100-200: Freeze first 10 ViT blocks  
freeze_config:
  vision_backbone_num_frozen_layers: 10

# Epoch 200+: Full fine-tuning
freeze_config:
  vision_backbone_num_frozen_layers: null
```
**Best for:** Transfer learning with careful adaptation

### Strategy 5: Adapter-Style Training
Only train specific adapter layers:
```yaml
freeze_config:
  freeze_all_except:
    - "*.adapter.*"
    - "mask_decoder.output_upscaling.*"
```
**Best for:** Extreme parameter efficiency, adding custom layers

## Pattern Syntax

Patterns use Unix glob syntax:
- `*` matches any characters within a component
- `.` separates hierarchical names
- `[0-5]` matches digits 0-5

### Examples

```yaml
custom_patterns:
  # Freeze first 6 ViT blocks
  - "backbone.vision_backbone.trunk.blocks.[0-5].*"
  
  # Freeze all bias terms
  - "*.bias"
  
  # Freeze specific layer
  - "backbone.vision_backbone.trunk.blocks.0.attn.*"

unfreeze_patterns:
  # Unfreeze last ViT block even if vision_backbone=true
  - "backbone.vision_backbone.trunk.blocks.23.*"
  
  # Unfreeze all layer norms
  - "*.layer_norm*"
  - "*.norm*"
```

## Logging

The trainer automatically logs:
- Total parameters
- Trainable parameters (count and percentage)
- Frozen parameters (count and percentage)
- Parameter groups breakdown

A file `frozen_params.txt` is saved to the log directory listing all frozen and trainable parameters.

## Example Training Commands

```bash
# Train with backbone frozen
python sam3/train/train.py \
  -c configs/mirror_finetune_with_freezing.yaml \
  --use-cluster 0 \
  --num-gpus 1

# Multi-GPU training
python sam3/train/train.py \
  -c configs/mirror_finetune_with_freezing.yaml \
  --use-cluster 0 \
  --num-gpus 4
```

## Notes

1. **Checkpoint Size**: Frozen parameters are not saved in checkpoints by default, reducing checkpoint size significantly.

2. **DDP Compatibility**: Set `find_unused_parameters: True` in distributed config when using freezing.

3. **Learning Rates**: Frozen parameters are automatically excluded from the optimizer. You don't need to set their LR to 0.

4. **Resume Training**: When resuming, the same freeze configuration is applied automatically.

5. **Combining Options**: You can combine multiple freezing strategies. The priority is:
   1. `unfreeze_patterns` (highest - always unfreezes)
   2. `freeze_all_except` (if set, ignores component flags)
   3. Component flags + `custom_patterns`

## Troubleshooting

### CUDA Out of Memory
Increase freezing to reduce trainable parameters:
```yaml
freeze_config:
  vision_backbone: true
  language_backbone: true
```

### Training Doesn't Converge
Try reducing freezing or increasing learning rate for trainable components:
```yaml
scratch:
  lr_transformer: 2e-4  # Increase decoder LR when backbone is frozen
```

### Some Gradients Are NaN
Check that you're not trying to compute gradients for frozen parameters. The trainer should handle this automatically.
