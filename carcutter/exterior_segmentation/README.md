# Exterior segmentation — inference package

Self-contained implementation of the exterior car-cutout pipeline: photo in, cutout + see-through
holes + windows + antenna out. See `ARCHITECTURE.md` for the design, the measured numbers, and the
deployment decisions still open.

## Install

```bash
uv pip install torch torchvision opencv-python-headless numpy scipy pillow \
               rfdetr segmentation-models-pytorch timm einops kornia
git clone git@github.com:carcutter/BiRefNet.git   # the PATCHED fork, not upstream
export BIREFNET_REPO=$PWD/BiRefNet
```

Upstream `ZhengPeng7/BiRefNet` will **not** load these checkpoints — it lacks the `BIREFNET_NHEADS`
patch that lets the decoder emit 3 channels.

## Weights

Four checkpoints (~2.1 GB), tracked with DVC in `carcutter/model-registry`:

```bash
dvc pull model_repository/exterior_segmentation_v2
```

## Use

```python
from carcutter.exterior_segmentation import ExteriorSegmenter
import numpy as np
from PIL import Image

seg = ExteriorSegmenter({
    "detector":     f"{W}/rfdetr_ca_unified_v1.pth",
    "outline":      f"{W}/birefnet_aspect_prod_bucket5.pt",
    "trihead":      f"{W}/birefnet_trihead_bucket.pt",
    "antenna_unet": f"{W}/unet_antenna_v3.pt",
})

out = seg.predict(np.array(Image.open("car.jpg").convert("RGB")))
alpha = ExteriorSegmenter.cutout_alpha(out)      # silhouette minus see-through gaps
```

`out` holds `outline`, `punchout`, `windows`, `antenna` (bool HxW), plus `car_box`, `bucket`,
`full_frame_fallback`, and `antenna_head` (reference only — see ARCHITECTURE.md §5).

Batch CLI:

```bash
python -m carcutter.exterior_segmentation.demo --images IN --out OUT --weights WEIGHTS_DIR
```

## Layout

| file | role |
|---|---|
| `buckets.py` | the 5 aspect buckets, routing, letterboxing, normalisation |
| `models.py` | the spliced BiRefNet, detector and UNet loaders |
| `pipeline.py` | `ExteriorSegmenter.predict` — detection, routing, inference, assembly |
| `demo.py` | folder-in / masks-out CLI |

Research and training code (data prep, training, the full eval suite) lives in
`carcutter/car_bbox_detector/`. That is not needed to run inference.
