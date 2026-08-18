#!/bin/bash
# Full dual-head bucketed training: one bucketed Swin-L pass -> outline(ch0)+tint(ch1).
# Warm-start from prod_bucket5, full data + trunk x3, full-ft 8ep. Persistent logs.
set -e
cd /home/rutger/work/cc_segmentation_sam3_clean
PY=/home/rutger/miniconda3/envs/cc_sam3/bin/python
export HF_HOME=carcutter/car_bbox_detector/birefnet/hf_cache
export HF_HUB_CACHE=$HF_HOME HUGGINGFACE_HUB_CACHE=$HF_HOME
export PYTHONPATH=.:carcutter/car_bbox_detector/birefnet/BiRefNet
export BIREFNET_DUAL_HEAD=1
LOGD=carcutter/car_bbox_detector/experiments/aspect_logs
echo "[dualhead] $(date) START"
$PY carcutter/car_bbox_detector/train_dualhead_bucket.py \
  --n-train 0 --oversample-trunk 3 --epochs 8 --bs 2 --lr 1e-5 --tag dualhead_bucket \
  > $LOGD/dualhead.log 2>&1
echo "[dualhead] $(date) DONE"
echo "DUALHEAD_COMPLETE" > $LOGD/dualhead.flag
