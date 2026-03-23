#!/usr/bin/env python3
"""
General Script to create train/val split for segmentation datasets.

Features:
- Configurable for any task (outline, holes, mirrors, etc.)
- Stratification by number of segmented regions
- Optional integration with CLIP classification results for vehicle type stratification
- Ensures validation set only contains images with non-empty masks
- Supports multiple batches
"""

import os
import json
import shutil
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from PIL import Image
import numpy as np
import cv2

# Minimum area for counting regions (in pixels)
MIN_REGION_AREA = 100

# Number of random samples to save per category in tmp folder
NUM_RANDOM_SAMPLES_TMP = 5


def count_regions_in_mask(mask_path, min_area=MIN_REGION_AREA):
    """
    Count the number of distinct regions in a binary mask using connected components.
    Filters out small noise components.
    
    Args:
        mask_path: Path to the mask image
        min_area: Minimum area in pixels for a component to be counted
        
    Returns:
        Number of distinct instances (connected components)
    """
    try:
        # Load mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return 0
        
        # Convert to binary (assuming 255 is foreground)
        binary_mask = (mask > 127).astype(np.uint8)
        
        # Apply morphological closing to connect nearby regions (fill small gaps)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        
        # Label connected components using OpenCV (8-connectivity)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )
        
        # Count only components larger than min_area (skip background which is label 0)
        num_regions = 0
        for i in range(1, num_labels):  # Start from 1 to skip background
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                num_regions += 1
        
        return num_regions
    except Exception as e:
        print(f"Error processing {mask_path}: {e}")
        return 0


def has_non_empty_mask(mask_path):
    """
    Check if a mask has any foreground pixels.
    
    Args:
        mask_path: Path to the mask image
        
    Returns:
        True if mask has foreground pixels, False otherwise
    """
    try:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return False
        return np.any(mask > 127)
    except Exception as e:
        print(f"Error checking mask {mask_path}: {e}")
        return False


def get_mask_path(batch_folder, task_name, mask_pattern, image_basename):
    """
    Construct mask path based on configuration.
    
    Args:
        batch_folder: Base folder for the batch
        task_name: Name of the task
        mask_pattern: Pattern for mask path with {task} placeholder
        image_basename: Image filename without extension
        
    Returns:
        Path to the mask file
    """
    # Replace {task} placeholder with actual task name
    mask_subdir = mask_pattern.format(task=task_name)
    mask_path = os.path.join(batch_folder, mask_subdir, image_basename + '.png')
    return mask_path


def get_image_mask_pairs(images_dir, masks_dir):
    """
    Get all image-mask pairs.
    
    Returns:
        List of tuples (image_path, mask_path, image_file)
    """
    pairs = []
    
    for image_file in os.listdir(images_dir):
        if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        # Construct mask path (assuming same name but .png extension)
        image_basename = os.path.splitext(image_file)[0]
        mask_file = image_basename + '.png'
        
        image_path = os.path.join(images_dir, image_file)
        mask_path = os.path.join(masks_dir, mask_file)
        
        if os.path.exists(mask_path):
            pairs.append((image_path, mask_path, image_file))
        else:
            # Silently skip images without masks (common for some batches)
            pass
    
    return pairs


def load_clip_classification(json_path):
    """
    Load CLIP classification results from JSON.
    
    Returns:
        Dictionary mapping (batch_name, filename) -> category
    """
    if not os.path.exists(json_path):
        print(f"WARNING: CLIP classification file not found: {json_path}")
        return {}
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    classification_map = {}
    
    for batch_name, batch_data in data.get("batches", {}).items():
        if batch_data is None:
            continue
        detailed_results = batch_data.get("detailed_results", {})
        for filename, result in detailed_results.items():
            classification_map[(batch_name, filename)] = result.get("category", "unknown")
    
    return classification_map


def stratified_split(data_by_strata, val_ratio=0.15, random_seed=42):
    """
    Perform stratified split based on strata (number of regions, vehicle type, etc.).
    
    Args:
        data_by_strata: Dictionary mapping strata_key -> list of samples
        val_ratio: Ratio of samples to use for validation
        random_seed: Random seed for reproducibility
        
    Returns:
        train_data, val_data (lists of samples)
    """
    np.random.seed(random_seed)
    
    train_data = []
    val_data = []
    
    for strata_key, samples in data_by_strata.items():
        # If only 1 image in this group, keep it in training
        if len(samples) == 1:
            print(f"  Strata '{strata_key}' has only 1 image, keeping in training")
            train_data.extend(samples)
            continue
        
        # Shuffle samples
        shuffled = samples.copy()
        np.random.shuffle(shuffled)
        
        # Calculate split point
        n_val = max(1, int(len(samples) * val_ratio))  # At least 1 sample in val if possible
        
        # Split
        val_data.extend(shuffled[:n_val])
        train_data.extend(shuffled[n_val:])
    
    return train_data, val_data


def save_special_cases_to_tmp(data_by_regions, output_dir, special_counts=[0], 
                              num_random_samples=5, task_name="task"):
    """
    Save images with specific region counts to a tmp folder.
    Also saves random samples from all groups for verification.
    
    Args:
        data_by_regions: Dictionary mapping num_regions -> list of samples
        output_dir: Base output directory
        special_counts: List of region counts to save completely (e.g., [0] for exclusions)
        num_random_samples: Number of random samples to save from other groups
        task_name: Name of the task for folder naming
    """
    tmp_dir = os.path.join(output_dir, f'tmp_special_cases_{task_name}')
    os.makedirs(tmp_dir, exist_ok=True)
    
    print(f"\nSaving samples to: {tmp_dir}")
    
    for num_regions in sorted(data_by_regions.keys()):
        samples = data_by_regions[num_regions]
        subdir = os.path.join(tmp_dir, f'{num_regions}_regions')
        os.makedirs(os.path.join(subdir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(subdir, 'masks'), exist_ok=True)
        
        # Determine which samples to save
        if num_regions in special_counts:
            # Save all samples for special counts
            samples_to_save = samples
            print(f"  Saving all {len(samples)} images with {num_regions} regions...")
        else:
            # Save random samples for other groups
            n_samples = min(num_random_samples, len(samples))
            np.random.seed(42)
            samples_to_save = np.random.choice(samples, n_samples, replace=False).tolist()
            print(f"  Saving {n_samples} random samples with {num_regions} regions (out of {len(samples)})...")
        
        for sample in samples_to_save:
            image_file = sample['image_file']
            image_basename = os.path.splitext(image_file)[0]
            
            # Source paths (stored in sample)
            src_image = sample['image_path']
            src_mask = sample['mask_path']
            
            # Destination paths
            dst_image = os.path.join(subdir, 'images', image_file)
            dst_mask = os.path.join(subdir, 'masks', image_basename + '.png')
            
            shutil.copy2(src_image, dst_image)
            shutil.copy2(src_mask, dst_mask)


def copy_files_to_roboflow_structure(train_data, val_data, output_dir):
    """
    Create Roboflow folder structure and copy files.
    
    Structure:
        output_dir/
            train/
                images/
                labels/
            val/
                images/
                labels/
            test/
                images/
                labels/
    """
    # Create directories
    splits = ['train', 'val', 'test']
    subdirs = ['images', 'labels']
    
    for split in splits:
        for subdir in subdirs:
            path = os.path.join(output_dir, split, subdir)
            os.makedirs(path, exist_ok=True)
    
    # Copy training data
    print(f"\nCopying {len(train_data)} training samples...")
    for sample in train_data:
        image_file = sample['image_file']
        image_basename = os.path.splitext(image_file)[0]
        
        # Source paths (stored in sample)
        src_image = sample['image_path']
        src_mask = sample['mask_path']
        
        # Destination paths
        dst_image = os.path.join(output_dir, 'train', 'images', image_file)
        dst_mask = os.path.join(output_dir, 'train', 'labels', image_basename + '.png')
        
        shutil.copy2(src_image, dst_image)
        shutil.copy2(src_mask, dst_mask)
    
    # Copy validation data
    print(f"Copying {len(val_data)} validation samples...")
    for sample in val_data:
        image_file = sample['image_file']
        image_basename = os.path.splitext(image_file)[0]
        
        # Source paths (stored in sample)
        src_image = sample['image_path']
        src_mask = sample['mask_path']
        
        # Destination paths
        dst_image = os.path.join(output_dir, 'val', 'images', image_file)
        dst_mask = os.path.join(output_dir, 'val', 'labels', image_basename + '.png')
        
        shutil.copy2(src_image, dst_image)
        shutil.copy2(src_mask, dst_mask)
    
    print(f"\nTest folder created (empty) at: {os.path.join(output_dir, 'test')}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Create a stratified train/val split for a segmentation dataset."
    )
    parser.add_argument("--raw-dir", required=True, help="Path to cropped raw images")
    parser.add_argument("--mask-dir", required=True, help="Path to cropped binary masks")
    parser.add_argument(
        "--output-dir", required=True,
        help="Output dataset root; produces train/, val/, test/ subdirs"
    )
    parser.add_argument("--task-name", default="trailer", help="Task name used in metadata filenames (default: trailer)")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation fraction (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--no-exclude-empty", action="store_true",
        help="Include images with empty masks in the split (by default they are excluded)"
    )
    args = parser.parse_args()

    task_name = args.task_name
    val_ratio = args.val_ratio
    random_seed = args.seed
    exclude_empty = not args.no_exclude_empty
    output_dir = args.output_dir
    date_stamp = datetime.now().strftime("%Y%m%d")
    json_output = os.path.join(output_dir, f"train_val_split_{date_stamp}.json")

    print("=" * 80)
    print(f"{task_name.upper()} DATASET TRAIN/VAL SPLIT")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Task: {task_name}")
    print(f"  Raw images dir: {args.raw_dir}")
    print(f"  Masks dir:      {args.mask_dir}")
    print(f"  Output dir:     {output_dir}")
    print(f"  Validation ratio: {val_ratio * 100:.0f}%")
    print(f"  Exclude empty masks: {exclude_empty}")
    print(f"  Random seed: {random_seed}")

    # Get all image-mask pairs
    print("\n" + "-" * 80)
    print("Step 1: Finding image-mask pairs...")
    print("-" * 80)

    if not os.path.exists(args.raw_dir):
        print(f"ERROR: Raw images directory not found: {args.raw_dir}")
        return
    if not os.path.exists(args.mask_dir):
        print(f"ERROR: Masks directory not found: {args.mask_dir}")
        return

    raw_pairs = get_image_mask_pairs(args.raw_dir, args.mask_dir)
    all_pairs = [(img_path, mask_path, img_file, "direct") for img_path, mask_path, img_file in raw_pairs]
    print(f"Found {len(all_pairs)} image-mask pairs")

    if len(all_pairs) == 0:
        print("\nERROR: No image-mask pairs found. Please check the directory paths.")
        return

    # Count regions in each image
    print("\n" + "-" * 80)
    print("Step 2: Analyzing masks and counting regions...")
    print("-" * 80)

    data_by_regions = defaultdict(list)
    data_by_strata = defaultdict(list)

    for i, (image_path, mask_path, image_file, batch_name) in enumerate(all_pairs):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(all_pairs)} images...")

        num_regions = count_regions_in_mask(mask_path)
        sample = {
            'image_file': image_file,
            'image_path': image_path,
            'mask_path': mask_path,
            'batch_name': batch_name,
            'num_regions': num_regions,
            'vehicle_type': 'unknown',
        }
        data_by_regions[num_regions].append(sample)
        data_by_strata[f"{num_regions}_regions"].append(sample)

    print(f"Processed all {len(all_pairs)} images")

    # Print statistics
    print("\n" + "-" * 80)
    print("Step 3: Dataset statistics")
    print("-" * 80)
    print(f"\nDistribution of {task_name} regions per image:")
    for num_regions in sorted(data_by_regions.keys()):
        count = len(data_by_regions[num_regions])
        percentage = (count / len(all_pairs)) * 100
        print(f"  {num_regions} region(s): {count:4d} images ({percentage:5.2f}%)")

    # Handle empty masks
    excluded_empty = []
    if exclude_empty:
        print("\n" + "-" * 80)
        print(f"Step 4: Excluding images with 0 {task_name} regions from split")
        print("-" * 80)
        excluded_empty = data_by_regions.pop(0, [])
        for key in [k for k in data_by_strata if k.startswith("0_regions")]:
            del data_by_strata[key]
        if excluded_empty:
            print(f"Excluded {len(excluded_empty)} images with 0 regions")
    else:
        print("\nStep 4: Including all images (including empty masks)")

    # Stratified split
    print("\n" + "-" * 80)
    print(f"Step 5: Performing stratified split ({val_ratio*100:.0f}% validation)")
    print("-" * 80)

    train_data, val_data = stratified_split(data_by_strata, val_ratio=val_ratio, random_seed=random_seed)

    # Verify val set has non-empty masks
    val_data_verified = []
    val_rejected = 0
    for sample in val_data:
        if has_non_empty_mask(sample['mask_path']):
            val_data_verified.append(sample)
        else:
            train_data.append(sample)
            val_rejected += 1
    if val_rejected > 0:
        print(f"  Moved {val_rejected} images with empty masks from validation to training")
    val_data = val_data_verified

    total_after_exclusion = len(train_data) + len(val_data)
    if total_after_exclusion == 0:
        print("\nERROR: No images available for split after exclusions.")
        return

    print(f"\nSplit results:")
    print(f"  Training set:   {len(train_data):4d} images ({len(train_data)/total_after_exclusion*100:.2f}%)")
    print(f"  Validation set: {len(val_data):4d} images ({len(val_data)/total_after_exclusion*100:.2f}%)")
    if excluded_empty:
        print(f"  Excluded:       {len(excluded_empty):4d} images with 0 regions")

    print(f"\n  Training set distribution by {task_name} regions:")
    train_by_regions = defaultdict(int)
    for sample in train_data:
        train_by_regions[sample['num_regions']] += 1
    for nr in sorted(train_by_regions.keys()):
        print(f"    {nr} region(s): {train_by_regions[nr]:4d} images")

    print(f"\n  Validation set distribution by {task_name} regions:")
    val_by_regions = defaultdict(int)
    for sample in val_data:
        val_by_regions[sample['num_regions']] += 1
    for nr in sorted(val_by_regions.keys()):
        print(f"    {nr} region(s): {val_by_regions[nr]:4d} images")

    # Save special cases to tmp folder
    print("\n" + "-" * 80)
    print("Step 6: Saving samples to tmp folder")
    print("-" * 80)
    if excluded_empty:
        data_by_regions[0] = excluded_empty
    save_special_cases_to_tmp(
        data_by_regions, output_dir,
        special_counts=[0], num_random_samples=NUM_RANDOM_SAMPLES_TMP,
        task_name=task_name
    )

    # Save JSON metadata
    print("\n" + "-" * 80)
    print("Step 7: Saving split information to JSON")
    print("-" * 80)

    split_info = {
        'task': task_name,
        'raw_dir': args.raw_dir,
        'mask_dir': args.mask_dir,
        'date': date_stamp,
        'total_images': len(all_pairs),
        'excluded_images': len(excluded_empty),
        'train_size': len(train_data),
        'val_size': len(val_data),
        'val_ratio': val_ratio,
        'random_seed': random_seed,
        'exclude_empty_masks': exclude_empty,
        'distribution_by_regions': {str(k): len(v) for k, v in sorted(data_by_regions.items())},
        'train': train_data,
        'val': val_data,
        'excluded_empty_masks': excluded_empty,
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(json_output, 'w') as f:
        json.dump(split_info, f, indent=2)
    print(f"Split information saved to: {json_output}")

    # Create Roboflow folder structure
    print("\n" + "-" * 80)
    print("Step 8: Creating Roboflow folder structure")
    print("-" * 80)

    copy_files_to_roboflow_structure(train_data, val_data, output_dir)

    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"\nRoboflow folder structure created at: {output_dir}")
    print(f"Split information saved at: {json_output}")
    print("\nFolder structure:")
    print(f"  {output_dir}/")
    print(f"    train/  images/ ({len(train_data)} images)  labels/ ({len(train_data)} masks)")
    print(f"    val/    images/ ({len(val_data)} images)  labels/ ({len(val_data)} masks)")
    print(f"    test/   images/ (empty)  labels/ (empty)")
    if excluded_empty:
        print(f"    tmp_special_cases_{task_name}/  0_regions/ ({len(excluded_empty)} excluded images)")


if __name__ == '__main__':
    main()
