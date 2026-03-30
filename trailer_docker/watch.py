#!/usr/bin/env python3
"""
Folder-watching PSD export service.

Polls /input for new images, processes them, writes PSDs to /output.

An image is skipped if a .psd with the same stem already exists in /output.
An image is held until its file size is stable (fully written / copied over
the network) before being included in a batch.

Images are batched together so both models are loaded only once per poll cycle.

Usage (Docker):
    docker run --gpus all -v /your/input:/input -v /your/output:/output trailer-seg

Usage (local):
    python watch.py --input-dir ./test_input --output-dir ./test_output \
                    --checkpoint /path/to/checkpoint.pt --device cuda
"""

import argparse
import logging
import sys
import tempfile
import time
from pathlib import Path

import torch

# Both watch.py and run_psd_export.py live in the same directory
# (in the container: /app/; locally: trailer_docker/).
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "inference"))

from run_psd_export import IMAGE_EXTENSIONS, run_psd_export

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CONTAINER_FINETUNED_CHECKPOINT = "/model/checkpoint.pt"
CONTAINER_BASE_CHECKPOINT     = "/model/sam3_base.pt"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def find_pending(input_dir: Path, output_dir: Path) -> list[Path]:
    """Images in input_dir that have no matching .psd in output_dir."""
    return sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
        and not (output_dir / (p.stem + ".psd")).exists()
    )


def is_stable(path: Path, wait: float) -> bool:
    """Return True if file size is unchanged over `wait` seconds and non-zero."""
    try:
        size_before = path.stat().st_size
        if size_before == 0:
            return False
        time.sleep(wait)
        return path.stat().st_size == size_before
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def watch(args: argparse.Namespace) -> None:
    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # CPU fallback
    if args.device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA not available — falling back to CPU (will be slower)")
        args.device = "cpu"

    log.info(f"Input     : {input_dir}")
    log.info(f"Output    : {output_dir}")
    log.info(f"Device    : {args.device}")
    log.info(f"Checkpoint: {args.checkpoint}")
    log.info(f"Poll every {args.poll_interval}s  stability check {args.stability_wait}s")
    log.info("Ready — waiting for images…")

    while True:
        pending = find_pending(input_dir, output_dir)

        if not pending:
            time.sleep(args.poll_interval)
            continue

        log.info(f"{len(pending)} pending — checking stability…")
        stable = [p for p in pending if is_stable(p, args.stability_wait)]

        if not stable:
            log.info("None stable yet, retrying next poll")
            time.sleep(args.poll_interval)
            continue

        log.info(f"Batch of {len(stable)}: {[p.name for p in stable]}")

        # Symlink stable files into a temp dir so run_psd_export sees a clean batch
        with tempfile.TemporaryDirectory(prefix="sam3_batch_") as tmp:
            tmp_path = Path(tmp)
            for p in stable:
                (tmp_path / p.name).symlink_to(p.resolve())

            try:
                run_psd_export(
                    input_dir=str(tmp_path),
                    output_dir=str(output_dir),
                    finetuned_checkpoint=args.checkpoint,
                    base_checkpoint=args.base_checkpoint,
                    device=args.device,
                    trailer_threshold=args.trailer_threshold,
                    base_threshold=args.base_threshold,
                    padding_frac=args.padding,
                    min_crop_px=args.min_crop_px,
                )
            except Exception as exc:
                log.error(f"Batch failed: {exc}", exc_info=True)

        time.sleep(args.poll_interval)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Watch a folder for trailer images and export PSDs."
    )
    parser.add_argument(
        "--input-dir", default="/input",
        help="Folder to watch for incoming images (default: /input)"
    )
    parser.add_argument(
        "--output-dir", default="/output",
        help="Folder where PSDs are written (default: /output)"
    )
    parser.add_argument(
        "--checkpoint", default=CONTAINER_FINETUNED_CHECKPOINT,
        help=f"Finetuned trailer checkpoint .pt (default: {CONTAINER_FINETUNED_CHECKPOINT})"
    )
    parser.add_argument(
        "--base-checkpoint", default=CONTAINER_BASE_CHECKPOINT,
        help=f"SAM3 base model checkpoint .pt (default: {CONTAINER_BASE_CHECKPOINT})"
    )
    parser.add_argument(
        "--device", default="cuda",
        help="cuda or cpu — cuda falls back to cpu automatically if unavailable (default: cuda)"
    )
    parser.add_argument(
        "--poll-interval", type=float, default=10.0,
        help="Seconds between directory scans (default: 10)"
    )
    parser.add_argument(
        "--stability-wait", type=float, default=3.0,
        help="Seconds to wait when checking if a file is fully written (default: 3)"
    )
    parser.add_argument("--trailer-threshold", type=float, default=0.5)
    parser.add_argument("--base-threshold",    type=float, default=0.3)
    parser.add_argument("--padding",           type=float, default=0.4)
    parser.add_argument("--min-crop-px",       type=int,   default=300)

    watch(parser.parse_args())


if __name__ == "__main__":
    main()
