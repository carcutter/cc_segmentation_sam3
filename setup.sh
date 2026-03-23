#!/usr/bin/env bash
# setup.sh — Create the cc_sam3 conda environment and install all dependencies.
#
# Usage:
#   bash setup.sh
#
# After setup, run the pipeline using the environment's Python directly:
#   /path/to/miniconda3/envs/cc_sam3/bin/python <script>
#
# Or activate the environment first:
#   conda activate cc_sam3
#   python <script>
#
# NOTE: `conda run -n cc_sam3 python ...` may pick up a different Python if a
# .venv is present in the working directory. Use the full Python path or
# activate the environment to avoid this.

set -e

ENV_NAME="cc_sam3"
PYTHON_VERSION="3.12"

echo "=== Creating conda environment: $ENV_NAME ==="
conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y

PYTHON="$(conda run -n "$ENV_NAME" which python 2>/dev/null || echo "$HOME/miniconda3/envs/$ENV_NAME/bin/python")"
# Fall back to finding it directly if conda run picks up wrong Python
if [ ! -f "$HOME/miniconda3/envs/$ENV_NAME/bin/python" ]; then
    CONDA_PREFIX=$(conda info --base)
    PYTHON="$CONDA_PREFIX/envs/$ENV_NAME/bin/python"
else
    PYTHON="$HOME/miniconda3/envs/$ENV_NAME/bin/python"
fi

echo "=== Using Python: $PYTHON ==="
echo "=== Installing PyTorch (CUDA 12.6) ==="
"$PYTHON" -m pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu126

echo "=== Installing sam3 package with train dependencies ==="
"$PYTHON" -m pip install -e ".[train,dev]"

echo "=== Installing submitit and open_clip_torch ==="
"$PYTHON" -m pip install submitit open_clip_torch

echo ""
echo "=== Setup complete! ==="
echo ""
echo "To run the pipeline:"
echo "  $PYTHON carcutter/data_preparation/extract_binary_mask.py --help"
echo "  $PYTHON sam3/train/train.py -c configs/trailer/freeze/trailer_finetune_with_freezing.yaml --use-cluster 0 --num-gpus 1"
echo ""
echo "Or activate the environment:"
echo "  conda activate $ENV_NAME"
