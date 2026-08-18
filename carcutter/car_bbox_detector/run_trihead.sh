#!/bin/bash
# Full 3-head bucketed BiRefNet: one pass -> outline(ch0)+tint(ch1)+antenna(ch2). Warm-start prod5,
# full data + trunk x3, full-ft 8ep. Persistent logs.
set -e
cd /home/rutger/work/cc_segmentation_sam3_clean
PY=/home/rutger/miniconda3/envs/cc_sam3/bin/python
export HF_HOME=carcutter/car_bbox_detector/birefnet/hf_cache HF_HUB_CACHE=$HF_HOME HUGGINGFACE_HUB_CACHE=$HF_HOME
export PYTHONPATH=.:carcutter/car_bbox_detector/birefnet/BiRefNet BIREFNET_NHEADS=3
LOGD=carcutter/car_bbox_detector/experiments/aspect_logs
echo "[trihead] $(date) START"
$PY carcutter/car_bbox_detector/train_trihead_bucket.py --n-train 0 --oversample-trunk 3 --epochs 8 --bs 2 --lr 1e-5 --freeze-encoder --tag trihead_bucket > $LOGD/trihead.log 2>&1
echo "[trihead] $(date) DONE"; echo "TRIHEAD_COMPLETE" > $LOGD/trihead.flag
