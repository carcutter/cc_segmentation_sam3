#!/bin/bash
# 5-BUCKET productionization: bucketed (P/A/B/C/D incl explicit portrait 0.91) + FULL data +
# open-trunk x3 oversampling + full-ft. Eval vs shipped car_outline_v2 by view. The probe's
# BUCKETS/route are now 5-way, so this trains/evals the 5-bucket model.
set -e
cd /home/rutger/work/cc_segmentation_sam3_clean
PY=/home/rutger/miniconda3/envs/cc_sam3/bin/python
export HF_HOME=carcutter/car_bbox_detector/birefnet/hf_cache
export HF_HUB_CACHE=$HF_HOME HUGGINGFACE_HUB_CACHE=$HF_HOME
export PYTHONPATH=.:carcutter/car_bbox_detector/birefnet/BiRefNet
LOGD=carcutter/car_bbox_detector/experiments/aspect_logs

echo "[prod5] $(date) START prod_bucket5 (5 buckets + full data + trunk x3)"
$PY carcutter/car_bbox_detector/probe_birefnet_aspect.py \
  --full-ft --n-train 0 --oversample-trunk 3 --epochs 8 --bs 2 --lr 1e-5 \
  --tag prod_bucket5 > $LOGD/prod_bucket5.log 2>&1
echo "[prod5] $(date) DONE"
echo "PROD5_COMPLETE" > $LOGD/aspect_prod5.flag
