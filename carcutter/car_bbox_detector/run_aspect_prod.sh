#!/bin/bash
# Productionization: the deployable bucketed outline model — FULL train data + open-trunk
# oversampling x3 (restores the car_outline_v2 fix that the probe subsample dropped) + bucketed
# aspect. Eval vs shipped car_outline_v2 by view. Launch only after the isolate verdict confirms
# bucketing wins/ties-with-side-gain.
set -e
cd /home/rutger/work/cc_segmentation_sam3_clean
PY=/home/rutger/miniconda3/envs/cc_sam3/bin/python
export HF_HOME=carcutter/car_bbox_detector/birefnet/hf_cache
export HF_HUB_CACHE=$HF_HOME HUGGINGFACE_HUB_CACHE=$HF_HOME
export PYTHONPATH=.:carcutter/car_bbox_detector/birefnet/BiRefNet

LOGD=carcutter/car_bbox_detector/experiments/aspect_logs   # persistent (/tmp gets cleaned)
echo "[prod] $(date) START prod_bucketed (full data + trunk x3)"
$PY carcutter/car_bbox_detector/probe_birefnet_aspect.py \
  --full-ft --n-train 0 --oversample-trunk 3 --epochs 8 --bs 2 --lr 1e-5 \
  --tag prod_bucketed > $LOGD/prod_bucketed.log 2>&1
echo "[prod] $(date) DONE"
echo "PROD_COMPLETE" > $LOGD/aspect_prod.flag
