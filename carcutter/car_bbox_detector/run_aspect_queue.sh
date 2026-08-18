#!/bin/bash
# Queue: isolate ASPECT cleanly — train bucketed and square through the IDENTICAL pipeline
# (same init=car_outline_v2, data=4000, epochs=12, aug, lr), only input geometry differs.
# Then compare square-mine vs bucketed-mine BF@1 by view. Sequential (one GPU).
set -e
cd /home/rutger/work/cc_segmentation_sam3_clean
PY=/home/rutger/miniconda3/envs/cc_sam3/bin/python
export HF_HOME=carcutter/car_bbox_detector/birefnet/hf_cache
export HF_HUB_CACHE=$HF_HOME HUGGINGFACE_HUB_CACHE=$HF_HOME
export PYTHONPATH=.:carcutter/car_bbox_detector/birefnet/BiRefNet
COMMON="--full-ft --n-train 4000 --epochs 12 --bs 2 --lr 1e-5"

echo "[queue] $(date) START bucketed"
$PY carcutter/car_bbox_detector/probe_birefnet_aspect.py $COMMON --tag bucketed       > /tmp/queue_bucketed.log 2>&1
echo "[queue] $(date) START square-control"
$PY carcutter/car_bbox_detector/probe_birefnet_aspect.py $COMMON --tag square --square > /tmp/queue_square.log 2>&1
echo "[queue] $(date) BOTH DONE"

# compare
$PY - <<'EOF'
import csv
def load(t):
    return {r["view"]:(int(r["n"]),float(r["prod_sq"]),float(r["this"])) for r in
            csv.DictReader(open(f"carcutter/car_bbox_detector/experiments/aspect_byview_{t}.csv"))}
b=load("bucketed"); s=load("square")
print("\n=== ISOLATE-ASPECT VERDICT: BF@1 by view (same pipeline) ===")
print(f"  {'view':<9}{'n':>5}{'prod_sq':>9}{'square':>9}{'bucketed':>10}{'buck-sq':>9}")
for v in ["straight","corner","corner34","side","other","ALL"]:
    if v in b and v in s:
        n,prod,bk=b[v]; _,_,sq=s[v]
        print(f"  {v:<9}{n:>5}{prod:>9.3f}{sq:>9.3f}{bk:>10.3f}{bk-sq:>+9.3f}")
EOF
echo "QUEUE_COMPLETE" > /tmp/aspect_queue.flag
