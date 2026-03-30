# Trailer segmentation PSD export service
#
# Build (run download_weights.py first):
#   docker build -t trailer-seg .
#
# Run:
#   docker run --gpus all -v /your/input:/input -v /your/output:/output trailer-seg

FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# System libs needed by opencv / numba / pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# All Python deps in one step (requirements.txt pins torch+cu126 via --extra-index-url)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install sam3 package (deps already installed above)
COPY sam3/       sam3/
COPY pyproject.toml .
RUN pip install --no-cache-dir --no-deps -e .

# Pipeline + watcher
COPY inference/run_psd_export.py /app/run_psd_export.py
COPY watch.py                    /app/watch.py

# Baked-in weights
COPY weights/sam3_base.pt  /model/sam3_base.pt
COPY weights/checkpoint.pt /model/checkpoint.pt

VOLUME ["/input", "/output"]

ENTRYPOINT ["python", "/app/watch.py"]
