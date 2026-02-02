# Carcutter

Carcutter is a tool for car image segmentation and processing using SAM3.

## Setup

### Prerequisites
- Python 3.12
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

1. **Create a virtual environment with Python 3.12:**
   ```bash
   uv venv --python 3.12
   ```

2. **Activate the virtual environment:**
   ```bash
   source /home/raul/workspace/cc_segmentation_sam3/.venv/bin/activate
   ```

3. **Install the parent package (sam3) in editable mode:**
   ```bash
   uv pip install -e .
   ```

4. **Navigate to the carcutter directory and install requirements:**
   ```bash
   cd carcutter
   uv pip install -r ./requirements.txt
   ```

## Usage

[Add usage instructions here]

## Project Structure

```
carcutter/
├── README.md
├── requirements.txt
└── [other files]
```
