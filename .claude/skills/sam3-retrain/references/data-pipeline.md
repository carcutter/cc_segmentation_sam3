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

## Annotation Color Map

RGB color labels used across all dataset types (car exterior, car interior, trailer, boat):

| RGB | Hex | Label | Notes |
|-----|-----|-------|-------|
| `(0, 0, 0)` | `#000000` | Background | Always excluded |
| `(255, 0, 0)` | `#FF0000` | Car/boat/trailer body | Main foreground object |
| `(0, 255, 0)` | `#00FF00` | Mirror | Exterior side mirrors |
| `(0, 0, 255)` | `#0000FF` | See-through region | Transparent/window area showing background |
| `(255, 255, 255)` | `#FFFFFF` | Antenna | Long antennas + small nubs |
| `(255, 199, 200)` | `#FFC7C8` | Front left wheel | |
| `(128, 128, 128)` | `#808080` | Back left wheel | |
| `(200, 0, 255)` | `#C800FF` | Back right wheel | |
| `(255, 165, 0)` | `#FFA500` | Front right wheel | |

### Per-scene notes

**Car exterior:** all colors above apply.

**Car interior:** Blue = see-through window, Red = car body, Green = mirror,
Black = stickers (visible from inside) — stickers should be **merged with blue** for
training (they are effectively transparent/unwanted regions).

**Boat interior:** same labels as car interior.

**Boat exterior:** same labels as car exterior.
- Large boats: trailer/support that obscures the boat body is annotated as part
  of the boat (red).
- Small boats on stilts: boat body only, support structure not annotated.

**Trailer:** same color scheme as car exterior. Wheel labels (`front-left`, etc.) may be
inconsistent when the trailer has more or fewer than 4 wheels.

### How to use with extract_binary_mask.py

| Task | Mode | Colors to extract |
|------|------|-------------------|
| `outline` | `all_except_background` | Everything except black (body + mirrors + antenna + wheels) |
| `holes` | `by_color_ids` | `(0, 0, 255)` — blue see-through only |
| `mirror` | `by_color_ids` | `(0, 255, 0)` — green mirrors only |
| `trailer` | `all_except_background` | Everything except black |
| `antenna` | `by_color_ids` | `(255, 255, 255)` — white antenna only |

```bash
# outline / trailer — extract all foreground
python carcutter/data_preparation/extract_binary_mask.py \
  --input-dir <masks> --output-dir <binary_masks> \
  --mode all_except_background

# holes — extract only blue see-through regions
python carcutter/data_preparation/extract_binary_mask.py \
  --input-dir <masks> --output-dir <binary_masks> \
  --mode by_color_ids --background-color 255,0,0
  # Note: by_color_ids keeps white (255,255,255) by default;
  # to target blue, patch category_ids in the script or use check_mask_colors.py first
```

Use `check_mask_colors.py` to verify which colors are actually present in a batch:
```bash
python carcutter/data_preparation/check_mask_colors.py --folder <masks_dir>
```

---

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
