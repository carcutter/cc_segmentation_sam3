# Data Pipeline Reference

## Expected dataset structure

```
data/training/{task_name}/{date}/
├── train/
│   ├── images/     # Training images
│   └── labels/     # Binary masks (white=object, black=background)
├── val/
│   ├── images/
│   └── labels/
└── sam3_format/
    └── annotations/
        ├── instances_train.json
        └── instances_val.json
```

## Script: extract_binary_mask.py

Location: `carcutter/data_preparation/`

Extracts binary (white/black) masks from colored RGB segmentation masks.

```bash
python carcutter/data_preparation/extract_binary_mask.py \
  --input-dir <colored_masks_dir> \
  --output-dir <binary_masks_dir> \
  --mode all_except_background   # or by_color_ids
  [--tolerance 10]               # color matching tolerance
  [--fill-holes]                 # fill enclosed holes (e.g. chain-link fences)
  [--generate-contours]          # also produce contour masks
```

## Script: crop_to_outline_bbox.py

Location: `carcutter/data_preparation/`

Crops raw images and masks to the bounding box of the foreground object (5% padding default).

```bash
python carcutter/data_preparation/crop_to_outline_bbox.py \
  --raw-dir <raw_images_dir> \
  --mask-dir <binary_masks_dir> \
  --output-dir <batch_dir>         # produces raw_foreground_crop/ and masks_foreground_crop/
  [--outline-dir <outline_dir>]    # separate outline mask for bbox; defaults to --mask-dir
  [--padding 0.05]
```

## Script: create_train_val_split_general.py

Location: `carcutter/data_preparation/prepare_splits/`

```bash
python carcutter/data_preparation/prepare_splits/create_train_val_split_general.py \
  --raw-dir <cropped_raw_dir> \
  --mask-dir <cropped_mask_dir> \
  --output-dir data/training/<task>/<task>_<date> \
  --task-name <task> \
  [--val-ratio 0.15] [--seed 42]
```

Features:
- Stratifies split by number of segmented regions
- Ensures validation set has non-empty masks
- Outputs metadata JSON: `train_val_split_{date}.json`

## Script: prepare_sam3_dataset.py

Location: `carcutter/data_preparation/prepare_sam3_format/`

```bash
python carcutter/data_preparation/prepare_sam3_format/prepare_sam3_dataset.py \
  --data-root data/training/<task>/<task>_<date> \
  --task-name <task> \
  --category-names <task> \
  --min-area 100 \
  --min-hole-area 50
```

Converts binary masks to COCO JSON with RLE encoding.
- Preserves holes in masks (critical for the holes task; `--min-hole-area 50` to preserve chain-link detail)
- Filters by `--min-area` and `--min-hole-area`
- Generates bbox from mask automatically

## COCO JSON format used

```json
{
  "images": [{"id": 1, "file_name": "img.jpg", "height": H, "width": W}],
  "annotations": [{
    "id": 1,
    "image_id": 1,
    "category_id": 1,
    "segmentation": {"counts": "RLE_STRING", "size": [H, W]},
    "area": 12345,
    "bbox": [x, y, w, h],
    "iscrowd": 0
  }],
  "categories": [{"id": 1, "name": "trailer", "supercategory": "object"}]
}
