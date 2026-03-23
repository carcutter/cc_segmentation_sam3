---
name: sam3-label-review
description: Guides label quality analysis and annotation review for SAM3 segmentation datasets. Use when asked to analyze label quality, find mislabeled images, review annotations, run the annotation review tool, or inspect GT vs prediction disagreements.
metadata:
  author: carcutter
  version: 1.0.0
  category: data-quality
  tags: [segmentation, sam3, label-quality, annotation-review]
---

# SAM3 Label Quality Analysis & Annotation Review

Two-phase pipeline: first run the model over the labeled dataset to find
disagreements, then use the web app to review them and mark bad labels.

---

## Phase 1: Label Quality Analysis

Script: `carcutter/data_preparation/analyze_label_quality.py`

Runs a trained SAM3 checkpoint over all images in a split, computes
per-image metrics (IoU, mislabeling score, genuine miss score), saves
prediction masks, and generates ranked visualizations of the worst images.

### Configure at the top of the script

```python
checkpoint_path   = "path/to/checkpoint.pt"
data_root         = "data/training/outline_foreground_crop/<date>"
splits            = ["val"]          # or ["train", "val"]
output_dir        = "experiments/<run>/label_quality_analysis"
text_prompt       = "vehicle"
top_n_visualize   = 50
max_images        = None             # set to e.g. 20 for a smoke test
```

### Run

```bash
/home/rutger/miniconda3/envs/sam3/bin/python \
    carcutter/data_preparation/analyze_label_quality.py
```

Takes ~25 min for 4000 images on a single GPU.

### Outputs

| Path | Description |
|------|-------------|
| `{output_dir}/results.csv` | Per-image metrics for all images |
| `{output_dir}/pred_masks/` | Binary prediction masks (needed by review tool) |
| `{output_dir}/top_mislabeled/` | Visualizations ranked by mislabeling score |
| `{output_dir}/top_genuine_misses/` | Visualizations ranked by genuine miss score |

### Re-generate visualizations without re-running inference

```bash
/home/rutger/miniconda3/envs/sam3/bin/python \
    carcutter/data_preparation/analyze_label_quality.py --vis-only
```

### Key metrics in results.csv

| Column | Description |
|--------|-------------|
| `iou` | Intersection over Union between GT and prediction |
| `mislabeling_score` | Fraction of FN pixels that are blue (window glass) — high = likely mislabeled GT |
| `genuine_miss_score` | Fraction of GT that is missed antenna/body — high = model genuinely failing |
| `max_diff_blob_frac` | Largest connected error region as fraction of GT area — used to sort review queue |

Color-based FN classification uses the colored segmentation masks:
- **Blue** `(0,0,255)` = window glass → mislabeling signal (model correctly ignores it)
- **White** `(255,255,255)` = antenna → genuine miss signal
- **Red** `(255,0,0)` = body → genuine miss signal

---

## Phase 2: Annotation Review

Script: `annotation_review/review_labels.py`

Flask web app that shows GT / Pred / Difference panels side by side, sorted
worst-first by `max_diff_blob_frac` (largest error blob relative to GT area).

### Run

```bash
/home/rutger/miniconda3/envs/sam3/bin/python annotation_review/review_labels.py \
    --images  data/training/outline_foreground_crop/<date>/val/images \
    --gt      data/training/outline_foreground_crop/<date>/val/labels \
    --pred    experiments/<run>/label_quality_analysis/pred_masks \
    --scores  experiments/<run>/label_quality_analysis/results.csv \
    --output  review_decisions_val.csv \
    --port    5050
```

Open **http://127.0.0.1:5050** in the browser.

### Keyboard shortcuts

| Key | Action |
|-----|--------|
| `G` | Correct → auto-advance |
| `W` | Wrong → stays on image (draw bounding boxes first) |
| `S` | Skip → auto-advance |
| `← / →` | Navigate prev / next |
| `B` | Toggle draw mode for bounding boxes |
| `Esc` | Undo last box |
| `T` | Toggle overlays |
| `↑ / ↓` or `+ / -` | Brightness up / down |
| `scroll` | Zoom |
| `drag` | Pan |
| `double-click` | Reset zoom |

### Bounding box annotation (draw mode)

Press **B** to enter draw mode (button turns orange). Click-drag to mark the
error region with a yellow box. Click an existing box to delete it. Multiple
boxes per image are supported. Boxes are saved to the output CSV alongside the
decision for use in second-pass labeling.

### Output CSV columns

| Column | Description |
|--------|-------------|
| `stem` | Image filename without extension |
| `decision` | `correct`, `wrong`, or `skip` |
| `bboxes` | JSON array of `{"x","y","w","h"}` in image pixel coords (empty if none) |
| `max_diff_blob_frac` | Sort key used to rank images |
| `iou` | IoU from Phase 1 inference |
| `mislabeling_score` | From Phase 1 |
| `genuine_miss_score` | From Phase 1 |
| `timestamp` | UTC time of last decision |

Decisions and boxes are saved after every interaction and reloaded on restart,
so it is safe to stop and resume the review session.

---

## Common Issues

**Black panels in the browser**
- Hard-refresh the page (Ctrl+Shift+R) after restarting the server

**Images in wrong sort order**
- Confirm `--pred` points to the `pred_masks/` subfolder, not the experiment root
- Shape mismatches between GT and pred are silently skipped (check terminal output)

**Counter not updating**
- The progress counter fetches from the server after each decision; a brief delay is normal

**Server startup is slow (~10s)**
- Normal: it computes diff-blob sizes for all images at startup before serving
