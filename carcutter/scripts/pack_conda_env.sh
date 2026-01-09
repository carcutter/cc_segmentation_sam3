# install conda
rm -rf ./miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-py312_25.7.0-2-Linux-x86_64.sh
bash Miniconda3-py312_25.7.0-2-Linux-x86_64.sh -p ./miniconda -b
eval "$(./miniconda/bin/conda shell.bash hook)"

# create conda environment
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda create -n pt python=3.12.3 -y
conda activate pt
conda install -c conda-forge conda-pack -y

# pre install step
export PYTHONNOUSERSITE=True
conda install -c conda-forge libstdcxx-ng=15 -y

# install PyTorch and diffusers
pip install tokenizers numpy Pillow opencv-python torch

# pack environment
rm -f pb_exec_env_model
conda pack -o model_repository/pipeline/env.tar.gz

# deactivate conda
conda deactivate

rm Miniconda3-py312_25.7.0-2-Linux-x86_64.sh
