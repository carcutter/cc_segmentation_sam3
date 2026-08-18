#!/bin/bash
# RESUME the 5-bucket prod from its last per-epoch checkpoint (was killed during ep2; ckpt = epoch 1).
# Preserves the resume-from ckpt, then trains the remaining 6 epochs from it (8 total).
set -e
cd /home/rutger/work/cc_segmentation_sam3_clean
PY=/home/rutger/miniconda3/envs/cc_sam3/bin/python
export HF_HOME=carcutter/car_bbox_detector/birefnet/hf_cache
export HF_HUB_CACHE=$HF_HOME HUGGINGFACE_HUB_CACHE=$HF_HOME
export PYTHONPATH=.:carcutter/car_bbox_detector/birefnet/BiRefNet
EXP=carcutter/car_bbox_detector/experiments; LOGD=$EXP/aspect_logs
RESUME=$EXP/birefnet_aspect_prod_bucket5_resumefrom.pt

cp -f $EXP/birefnet_aspect_prod_bucket5.pt $RESUME    # preserve the epoch-1 weights
echo "[prod5-resume] $(date) START from $(basename $RESUME), 6 remaining epochs"
$PY carcutter/car_bbox_detector/probe_birefnet_aspect.py \
  --init $RESUME --full-ft --n-train 0 --oversample-trunk 3 --epochs 6 --bs 2 --lr 1e-5 \
  --tag prod_bucket5 > $LOGD/prod_bucket5_resume.log 2>&1
echo "[prod5-resume] $(date) DONE"
echo "PROD5_COMPLETE" > $LOGD/aspect_prod5.flag
