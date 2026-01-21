# SAM3 Dataset Preparation

This toolkit prepares your segmentation dataset for SAM3 finetuning by converting binary masks to COCO format with proper polygon annotations.

## Quick Start

**With date-stamped folders (recommended):**
```bash
python prepare_sam3_dataset.py \
    --data-root /path/to/data/training/<task_name>/<date> \
    --task-name <task_name>
```

**Auto-detect latest date folder:**
```bash
python prepare_sam3_dataset.py \
    --data-root /path/to/data/training/<task_name> \
    --task-name <task_name>
# Script will automatically use the latest date folder (e.g., 20260120)
```

**Example for mirror detection:**
```bash
python prepare_sam3_dataset.py \
    --data-root /home/raul/workspace/data/training/mirrors/20260120 \
    --task-name mirror
```

This will create COCO annotations at:
- `{data-root}/sam3_format/annotations/instances_train.json`
- `{data-root}/sam3_format/annotations/instances_val.json`

## Dataset Structure

**Structure with date folders (recommended):**
```
/path/to/data/training/<task_name>/<date>/
├── train/
│   ├── images/              # Training images
│   │   ├── image1.jpg
│   │   ├── image2.png
│   │   └── ...
│   └── labels/              # Training masks (white = object)
│       ├── image1.png
│       ├── image2.png
│       └── ...
├── val/
│   ├── images/              # Validation images
│   │   ├── image5.jpg
│   │   └── ...
│   └── labels/              # Validation masks
│       ├── image5.png
│       └── ...
├── test/ (optional)
│   ├── images/
│   └── labels/
├── tmp_special_cases/       # Special cases for review
└── train_val_split_<date>.json  # Split metadata
```

**Example for mirror detection:**
```
/home/raul/workspace/data/training/mirrors/20260120/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
├── tmp_special_cases/
└── train_val_split_20260120.json
```

**Output:**
```
/path/to/data/training/<task_name>/<date>/sam3_format/
└── annotations/
    ├── instances_train.json     # COCO format annotations for training
    └── instances_val.json       # COCO format annotations for validation

# Images remain at their original locations unless --copy-images is used
```


## Usage

### 1. Prepare Dataset

Basic usage with date folder (RECOMMENDED):
```bash
python prepare_sam3_dataset.py \
    --data-root /path/to/data/training/<task_name>/<date> \
    --task-name <task_name>
```

**Example for mirror detection:**
```bash
python prepare_sam3_dataset.py \
    --data-root /home/raul/workspace/data/training/mirrors/20260120 \
    --task-name mirror
```

**Auto-detect latest date:**
```bash
python prepare_sam3_dataset.py \
    --data-root /home/raul/workspace/data/training/mirrors \
    --task-name mirror
# Script will automatically find and use the latest date folder
```

This assumes your data structure is:
- Training images: `{data-root}/train/images`
- Training labels: `{data-root}/train/labels`
- Validation images: `{data-root}/val/images`
- Validation labels: `{data-root}/val/labels`

With additional options:
```bash
python prepare_sam3_dataset.py \
    --data-root /path/to/data/training/<task_name> \
    --task-name <task_name> \
    --output-dir /path/to/output \
    --category-name custom_name \
    --min-area 100
```

With fully custom paths for each directory:
```bash
python prepare_sam3_dataset.py \
    --train-images path/to/train/images \
    --train-labels path/to/train/labels \
    --val-images path/to/val/images \
    --val-labels path/to/val/labels \
    --output-dir workspace/data/mirrors/sam3_format
```

Copy images to output directory (optional):
```bash
python prepare_sam3_dataset.py --copy-images
```

**Parameters:**
- `--data-root`: Root directory for the task (can include or exclude date folder)
- `--date`: Specific date folder to use (e.g., 20260120). Auto-detects latest if not specified.
- `--task-name`: Name of the task (e.g., mirror, license_plate)
- `--train-images`: Override path to training images directory
- `--train-labels`: Override path to training labels directory
- `--val-images`: Override path to validation images directory
- `--val-labels`: Override path to validation labels directory
- `--output-dir`: Where to save the COCO annotations
- `--category-names`: List of category names (text prompts) for multi-prompt training
- `--min-area`: Minimum area in pixels to filter noise (default: 100)
- `--copy-images`: Copy images to output directory instead of keeping them in place

### 2. Visualize Annotations (Optional but Recommended)

Verify your annotations are correct:

```bash
# Visualize 10 random training samples
python visualize_coco_annotations.py \
    --dataset-dir /path/to/data/training/<task_name>/sam3_format \
    --split train \
    --num-samples 10
```

If images weren't copied, specify their location:
```bash
# Example for mirrors
python visualize_coco_annotations.py \
    --dataset-dir /home/raul/workspace/data/training/mirrors/sam3_format \
    --images-dir /home/raul/workspace/data/training/mirrors/train/images \
    --split train \
    --num-samples 10
```

Save visualizations to disk:
```bash
python visualize_coco_annotations.py \
    --dataset-dir /path/to/data/training/<task_name>/sam3_format \
    --split train \
    --num-samples 10 \
    --output-dir /path/to/visualizations
```

**Parameters:**
- `--dataset-dir`: Root directory containing annotations
- `--images-dir`: Override images directory (auto-detected if not specified)
- `--split`: Which split to visualize (`train` or `val`)
- `--num-samples`: Number of random samples to show
- `--output-dir`: Save visualizations instead of displaying (optional)
- `--no-bbox`: Don't show bounding boxes
- `--no-polygon`: Don't show polygon segmentations



## Key Features

### Works with Existing Train/Val Splits
The script reads from your existing training and validation splits, preserving your data organization.

### Automatic Instance Detection
The script automatically detects multiple object instances in each image and creates separate annotations for each.

### Polygon Annotations
Binary masks are converted to polygon contours, which SAM3 uses for training. The polygons are simplified to reduce memory usage while maintaining accuracy.

### Quality Filtering
- Minimum area threshold filters out noise and small artifacts
- Contour approximation reduces polygon complexity
- Invalid polygons (< 3 points) are automatically skipped

### Flexible Image Handling
- **Default**: Images stay at original locations (saves disk space)
- **Optional**: Copy images to output directory with `--copy-images` flag

### Task-Agnostic Design
- Works with any segmentation task (mirrors, license plates, cars, etc.)
- Category name automatically derived from task name or customizable
- Flexible directory structure support

## COCO Format Details

The generated annotations follow the standard COCO format:

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "image1.jpg",
      "height": 1024,
      "width": 768
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "segmentation": [[x1, y1, x2, y2, ...]],  // Polygon coordinates
      "area": 12345.67,
      "bbox": [x, y, width, height],
      "iscrowd": 0
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "<task_name>",
      "supercategory": "object"
    }
  ]
}
```

## Troubleshooting

### Required parameters missing
- Ensure you provide both `--data-root` and `--task-name`
- Example: `python prepare_sam3_dataset.py --data-root /path/to/data --task-name mirror`

### Directory not found errors
- Default structure expects `{data-root}/train/images`, `{data-root}/train/labels`, etc.
- Use `--train-images`, `--train-labels`, `--val-images`, `--val-labels` to specify custom paths
- Check that all directories exist and are readable

### No matching mask found
- Ensure masks have the same basename as images
- Check that mask files use common extensions (.png, .jpg, etc.)
- Verify the labels directory path is correct

### Too few instances detected
- Decrease `--min-area` to detect smaller objects
- Check that your masks have white (255) regions on black (0) background
- Verify masks are not inverted

### Polygons look incorrect
- Check the mask quality - blurry edges cause poor polygon extraction
- Consider pre-processing masks with morphological operations
- Adjust the contour approximation epsilon in the code if needed

### Images not found during visualization
- If images weren't copied, use `--images-dir` to specify their location
- Check that the file names in annotations match actual image files
- Verify paths are correct (relative vs absolute)

### Out of memory during training
- Reduce batch size in the SAM3 config
- Use gradient accumulation
- Consider smaller images or fewer instances per batch

## Advanced Usage

### Custom Filtering
Modify `extract_polygons_from_mask()` to add custom filtering:
```python
# Example: Filter by aspect ratio
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h
    if aspect_ratio < 0.3 or aspect_ratio > 3.0:
        continue  # Skip elongated objects
```

### Multi-Category Dataset
If you have multiple object types, modify the script to handle multiple categories:
```python
categories = [
    {"id": 1, "name": "mirror", "supercategory": "object"},
    {"id": 2, "name": "window", "supercategory": "object"},
]
```

### RLE Format (for very complex masks)
For masks with many holes or complex shapes, consider using RLE format instead of polygons. Modify the annotation creation:
```python
from pycocotools import mask as mask_utils

rle = mask_utils.encode(np.asfortranarray(binary_mask))
annotation["segmentation"] = rle
```

## Performance Tips

- **GPU**: Ensure CUDA is available for SAM3 training
- **Batch Size**: Start with smaller batches and increase gradually
- **Data Augmentation**: Consider adding augmentation in the SAM3 config
- **Learning Rate**: Start with 1e-4 and adjust based on validation loss
- **Checkpointing**: Save checkpoints frequently during training

## Complete Example: Mirror Detection

### Your Setup
```
/home/raul/workspace/data/training/mirrors/20260120/
├── train/images/
├── train/labels/
├── val/images/
├── val/labels/
├── tmp_special_cases/
└── train_val_split_20260120.json
```

### Step 1: Prepare Dataset
```bash
cd /home/raul/workspace/cc_segmentation_sam3/carcutter/data_preparation/prepare_sam3_format

# Option 1: Specify full path with date
python prepare_sam3_dataset.py \
    --data-root /home/raul/workspace/data/training/mirrors/20260120 \
    --task-name mirror

# Option 2: Auto-detect latest date folder
python prepare_sam3_dataset.py \
    --data-root /home/raul/workspace/data/training/mirrors \
    --task-name mirror
```

### Step 2: Verify Annotations
```bash
python verify_annotations.py \
    --dataset-dir /home/raul/workspace/data/training/mirrors/20260120/sam3_format \
    --splits train val
```

### Step 3: Visualize Samples
```bash
python visualize_coco_annotations.py \
    --dataset-dir /home/raul/workspace/data/training/mirrors/20260120/sam3_format \
    --images-dir /home/raul/workspace/data/training/mirrors/20260120/train/images \
    --split train \
    --num-samples 10
```

### Step 4: Train SAM3
Update your SAM3 config to point to the dataset:
```yaml
paths:
  dataset_root: /home/raul/workspace/data/training/mirrors/20260120

dataset:
  train_ann_file: sam3_format/annotations/instances_train.json
  val_ann_file: sam3_format/annotations/instances_val.json
  train_img_dir: train/images
  val_img_dir: val/images
```

Then run:
```bash
cd /path/to/sam3
python sam3/train/train.py -c configs/mirror_finetune.yaml
```

## References

- [SAM3 Repository](https://github.com/facebookresearch/sam3)
- [SAM3 Training Documentation](https://github.com/facebookresearch/sam3/blob/main/README_TRAIN.md)
- [COCO Dataset Format](https://cocodataset.org/#format-data)
