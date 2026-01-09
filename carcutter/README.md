# SAM3 - Car Segmentation

A Python implementation for car segmentation using SAM3 (Segment Anything Model 3), Meta AI's state-of-the-art segmentation model. This project provides inference capabilities for segmenting cars in images using both PyTorch and ONNX runtime, with support for Triton Inference Server deployment.

## Overview

SAM3 is a powerful segmentation model that can segment objects in images based on various prompts (boxes, points, text). This repository focuses on car segmentation tasks and includes:

- **PyTorch inference** scripts for direct model usage
- **ONNX runtime** support for optimized inference
- **Triton Inference Server** integration for production deployments
- Support for multiple input formats (boxes, points, text prompts)

## Features

- 🚗 Car segmentation using SAM3 model
- 🔥 PyTorch and ONNX inference support
- 🚀 Triton Inference Server integration
- 📦 Model export utilities (ONNX, TensorRT)

## Requirements

- Python 3.8+
- [uv](https://docs.astral.sh/uv/) package manager
- CUDA-capable GPU (recommended for optimal performance)

## Installation

This project uses [uv](https://docs.astral.sh/uv/), a fast Rust-based Python package manager, for dependency management.

### 1. Install uv

If you don't have `uv` installed, you can install it using:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone the Repository

```bash
git clone <repository-url>
cd sam3
```

### 3. Install Dependencies

Using the `uv.lock` file ensures reproducible installations with exact dependency versions:

# Install dependencies from the lock file

uv sync

### 4. Install Pre-commit Hooks

This repository uses pre-commit hooks to ensure code quality, security, and consistency. Install them with:

```bash
# Install pre-commit (if not already installed)
uv pip install pre-commit

# Install the git hooks
pre-commit install
```

The pre-commit hooks will automatically run on:

- **Pre-commit**: Before each commit (code formatting, linting, security checks)
- **Commit-msg**: Validating commit message format
- **Post-checkout/merge/rewrite**: After git operations

You can manually run all hooks on all files:

```bash
pre-commit run --all-files
```

### 5. HF Access

This model requires to be logged in with HuggingFace and to be accepted by the authors of the repo. Please make sure to authenticate with the HuggingFace CLI with

```bash
hf auth login
```

and if you want to modify the location of the model cache, set the `HF_HUB_CACHE` environment variable.

## Usage

### Basic Inference

Run inference on an image:

```bash
python inference_torch.py --image path/to/image.jpg --output output/
```

or use the Justfile

```bash
just run_inference
```

### ONNX Export

To export the model to ONNX format, run the provided export script:

```bash
python export_to_onnx.py
```

This will create the ONNX model files required for ONNX-based inference.

### ONNX Inference

For optimized ONNX inference:

```bash
python onnx_inference.py --image path/to/image.jpg --output output/
```

or use the Justfile

```bash
just run_inference_onnx
```

### Triton Inference Server

#### Prepare model repository

Triton Server expects a very specific folder structure, organized as follows:

```
model_repository/
├── vision-encoder/
│   └── 1/
│       └── model.onnx
│       └── model.plan
│   └── config.pbtxt
├── text-encoder/
│   └── 1/
│       └── model.onnx
│       └── model.plan
│   └── config.pbtxt
├── geometry-encoder/
│   └── 1/
│       └── model.onnx
│       └── model.plan
│   └── config.pbtxt
├── decoder/
│   └── 1/
│       └── model.onnx
│       └── model.plan
│   └── config.pbtxt
├── pipeline/
|   └── 1/
│       └── model.py
│       └── tokenizer.json
│   └── config.pbtxt
│   └── env.tar.gz
```

- Each folder in `model_repository` represents one model in the pipeline.
- The `1/` directory contains the versioned model or tokenizer file for Triton.
- Each model folder contains a `config.pbtxt` configuration file required by Triton.
- The `pipeline` folder also contains the env.tar.gz file, which contains the packed conda environment for the Python backend.

Launch the Triton Inference server instance with the command

```bash
just start_triton_server
```

Then, you can test the served models in two ways

1. Call the pipeline that internally calls the single models and orchestrates the inference. This approach will be used in production as is completely transparent to the user.

```bash
python call_triton_pipeline.py --image path/to/image.jpg
```

2. Call the models singularly and manage locally the orchestration. This is useful for debugging.

```bash
python call_triton_models.py --image path/to/image.jpg
```
