#!/usr/bin/env python3
"""
Crop raw images and masks to the bounding box of the outline (foreground) segmentation.
Background remains visible within the cropped region.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


def find_matching_image(raw_dir, mask_file, image_extensions):
    """
    Find matching raw image for a given mask file.
    Tries exact filename first, then matches by stem across extensions.
    """
    raw_path_exact = Path(raw_dir) / mask_file.name
    if raw_path_exact.exists():
        return raw_path_exact

    stem = mask_file.stem
    for ext in image_extensions:
        candidate = Path(raw_dir) / f"{stem}{ext}"
        if candidate.exists():
            return candidate

    return None


def find_matching_mask(mask_dir, outline_file, image_extensions):
    """
    Find matching mask file for a given outline file.
    Tries exact filename first, then matches by stem across extensions.
    """
    mask_path_exact = Path(mask_dir) / outline_file.name
    if mask_path_exact.exists():
        return mask_path_exact

    stem = outline_file.stem
    for ext in image_extensions:
        candidate = Path(mask_dir) / f"{stem}{ext}"
        if candidate.exists():
            return candidate

    return None


def get_foreground_bbox(mask, threshold=0, padding_pct=0.05):
    """
    Get bounding box of foreground pixels in mask.

    Args:
        mask: Single-channel mask (0 or 255)
        threshold: Threshold for foreground (default 0)
        padding_pct: Padding as percentage of bbox dimensions (default 0.05 for 5%)

    Returns:
        Tuple (x, y, w, h) or None if no foreground found
    """
    if len(mask.shape) == 3:
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask

    fg = mask_gray > threshold
    
    if not np.any(fg):
        return None

    # Find coordinates of foreground pixels
    coords = np.argwhere(fg)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Calculate padding based on bbox dimensions
    bbox_w = x_max - x_min + 1
    bbox_h = y_max - y_min + 1
    padding_x = int(bbox_w * padding_pct)
    padding_y = int(bbox_h * padding_pct)

    # Add padding
    h, w = mask_gray.shape
    x_min = max(0, x_min - padding_x)
    y_min = max(0, y_min - padding_y)
    x_max = min(w - 1, x_max + padding_x)
    y_max = min(h - 1, y_max + padding_y)

    return x_min, y_min, x_max - x_min + 1, y_max - y_min + 1


def crop_image_to_bbox(img, bbox):
    """
    Crop image to bounding box.

    Args:
        img: Image to crop
        bbox: Tuple (x, y, w, h)

    Returns:
        Cropped image
    """
    x, y, w, h = bbox
    return img[y:y+h, x:x+w]


def process_crop_to_outline_bbox(raw_dir, mask_dir, outline_dir, 
                                  output_raw_dir, output_mask_dir,
                                  threshold=0, padding_pct=0.05):
    """
    Crop raw images and masks to the bounding box of outline segmentation.

    Args:
        raw_dir: Directory containing raw images
        mask_dir: Directory containing class masks
        outline_dir: Directory containing outline masks (binary)
        output_raw_dir: Directory to save cropped raw images
        output_mask_dir: Directory to save cropped masks
        threshold: Foreground threshold for outline mask (default 0)
        padding_pct: Padding as percentage of bbox dimensions (default 0.05 for 5%)
    """
    os.makedirs(output_raw_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    outline_path = Path(outline_dir)
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    outline_files = []
    for ext in image_extensions:
        outline_files.extend(outline_path.glob(f'*{ext}'))

    outline_files = sorted(outline_files)

    if len(outline_files) == 0:
        print(f"No outline files found in {outline_dir}")
        return

    print(f"Found {len(outline_files)} outline files")
    print(f"Processing raw images from: {raw_dir}")
    print(f"Using masks from: {mask_dir}")
    print(f"Using outlines from: {outline_dir}")
    print(f"Saving cropped raw images to: {output_raw_dir}")
    print(f"Saving cropped masks to: {output_mask_dir}")

    stats = {
        'total': len(outline_files),
        'missing_raw': 0,
        'missing_mask': 0,
        'no_foreground': 0,
        'size_mismatch': 0,
        'processed': 0
    }

    for outline_file in tqdm(outline_files, desc="Cropping to outline bbox"):
        # Find matching raw image
        raw_path = find_matching_image(raw_dir, outline_file, image_extensions)
        if raw_path is None:
            stats['missing_raw'] += 1
            continue

        # Find matching mask
        mask_path = find_matching_mask(mask_dir, outline_file, image_extensions)
        if mask_path is None:
            stats['missing_mask'] += 1
            continue

        # Read images
        raw_img = cv2.imread(str(raw_path))
        if raw_img is None:
            stats['missing_raw'] += 1
            continue

        mask_img = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask_img is None:
            stats['missing_mask'] += 1
            continue

        outline_mask = cv2.imread(str(outline_file), cv2.IMREAD_UNCHANGED)
        if outline_mask is None:
            stats['no_foreground'] += 1
            continue

        # Check size consistency
        if raw_img.shape[:2] != outline_mask.shape[:2] or raw_img.shape[:2] != mask_img.shape[:2]:
            stats['size_mismatch'] += 1
            continue

        # Get bounding box from outline
        bbox = get_foreground_bbox(outline_mask, threshold=threshold, padding_pct=padding_pct)
        if bbox is None:
            stats['no_foreground'] += 1
            continue

        # Crop both raw image and mask
        cropped_raw = crop_image_to_bbox(raw_img, bbox)
        cropped_mask = crop_image_to_bbox(mask_img, bbox)

        # Save cropped images
        output_raw_path = Path(output_raw_dir) / raw_path.name
        output_mask_path = Path(output_mask_dir) / mask_path.name
        
        cv2.imwrite(str(output_raw_path), cropped_raw)
        cv2.imwrite(str(output_mask_path), cropped_mask)
        
        stats['processed'] += 1

    print(f"\n{'='*60}")
    print("Processing complete!")
    print(f"Total outlines processed: {stats['total']}")
    print(f"Processed image pairs: {stats['processed']}")
    print(f"Missing raw images: {stats['missing_raw']}")
    print(f"Missing masks: {stats['missing_mask']}")
    print(f"No foreground found: {stats['no_foreground']}")
    print(f"Size mismatches: {stats['size_mismatch']}")
    print(f"{'='*60}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Crop raw images and masks to the foreground bounding box."
    )
    parser.add_argument("--raw-dir", required=True, help="Path to raw images")
    parser.add_argument(
        "--mask-dir", required=True,
        help="Path to binary masks to crop (also used for bbox detection if --outline-dir is not set)"
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Parent output directory; produces raw_foreground_crop/ and masks_foreground_crop/ subdirs"
    )
    parser.add_argument(
        "--outline-dir", default=None,
        help="Path to binary outline masks used for bbox detection (defaults to --mask-dir)"
    )
    parser.add_argument(
        "--padding", type=float, default=0.05,
        help="Bounding box padding as a fraction of bbox dimensions (default: 0.05)"
    )
    args = parser.parse_args()

    outline_dir = args.outline_dir if args.outline_dir else args.mask_dir
    output_raw_dir = str(Path(args.output_dir) / "raw_foreground_crop")
    output_mask_dir = str(Path(args.output_dir) / "masks_foreground_crop")

    process_crop_to_outline_bbox(
        raw_dir=args.raw_dir,
        mask_dir=args.mask_dir,
        outline_dir=outline_dir,
        output_raw_dir=output_raw_dir,
        output_mask_dir=output_mask_dir,
        threshold=0,
        padding_pct=args.padding,
    )


if __name__ == '__main__':
    main()
