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

# ========================= CONFIGURATION =========================

# Task configuration
TASK_NAME = "holes"  # Change to: "outline", "holes", "mirrors", etc.

# Batch configuration
ROOT_FOLDER = "/home/raul/workspace/data"
BATCH_NAME_LIST = [
    "car_segmentation/kw2551_car_segmentation",
    "car_segmentation/kw2552_car_interior_segmentation",
    "car_segmentation/kw2553_car_segmentation",
    "car_segmentation/kw2602_car_segmentation"
]

# Mask path pattern (relative to batch folder)
# Use {task} as placeholder for task name
# Examples:
#   - "preprocessed_mask/{task}/mask" -> for outline, holes tasks
#   - "preprocessed_mask/mask" -> for mirrors task (no task subfolder)
MASK_PATH_PATTERN = "preprocessed_mask/{task}/mask"

# Output configuration
OUTPUT_BASE_DIR = "/home/raul/workspace/data/training"

# Split configuration
VAL_RATIO = 0.15  # 15% validation
RANDOM_SEED = 42

# CLIP classification integration (optional)
USE_CLIP_CLASSIFICATION = True  # Set to True to stratify by vehicle type
CLIP_CLASSIFICATION_JSON = "/home/raul/workspace/data/car_segmentation/classification_results/classification_results.json"

# Minimum area for counting regions (in pixels)
MIN_REGION_AREA = 100

# Whether to exclude images with empty masks (0 regions)
EXCLUDE_EMPTY_MASKS = True

# Number of random samples to save per category in tmp folder
NUM_RANDOM_SAMPLES_TMP = 5

# =================================================================


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
    # Generate date stamp for output
    date_stamp = datetime.now().strftime("%Y%m%d")
    
    # Output directories with date stamp
    output_dir = f'{OUTPUT_BASE_DIR}/{TASK_NAME}/{TASK_NAME}_{date_stamp}'
    json_output = f'{output_dir}/train_val_split_{date_stamp}.json'
    
    print("=" * 80)
    print(f"{TASK_NAME.upper()} DATASET TRAIN/VAL SPLIT")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Task: {TASK_NAME}")
    print(f"  Mask pattern: {MASK_PATH_PATTERN}")
    print(f"  Validation ratio: {VAL_RATIO * 100:.0f}%")
    print(f"  Exclude empty masks: {EXCLUDE_EMPTY_MASKS}")
    print(f"  Use CLIP classification: {USE_CLIP_CLASSIFICATION}")
    print(f"\nBatches to process: {len(BATCH_NAME_LIST)}")
    for batch_name in BATCH_NAME_LIST:
        print(f"  - {batch_name}")
    print(f"\nOutput directory: {output_dir}")
    print(f"JSON output: {json_output}")
    
    # Load CLIP classification if enabled
    clip_classification = {}
    if USE_CLIP_CLASSIFICATION:
        print("\n" + "-" * 80)
        print("Loading CLIP classification results...")
        print("-" * 80)
        clip_classification = load_clip_classification(CLIP_CLASSIFICATION_JSON)
        if clip_classification:
            print(f"Loaded classification for {len(clip_classification)} images")
        else:
            print("WARNING: No CLIP classification data loaded, continuing without vehicle type stratification")
    
    # Get all image-mask pairs from all batches
    print("\n" + "-" * 80)
    print("Step 1: Finding image-mask pairs from all batches...")
    print("-" * 80)
    
    all_pairs = []
    for batch_name in BATCH_NAME_LIST:
        folder_path = f"{ROOT_FOLDER}/{batch_name}"
        images_dir = f"{folder_path}/raw"
        
        # Construct mask directory based on pattern
        mask_subdir = MASK_PATH_PATTERN.format(task=TASK_NAME)
        masks_dir = f"{folder_path}/{mask_subdir}"
        
        print(f"\n  Processing batch: {batch_name}")
        print(f"    Images: {images_dir}")
        print(f"    Masks: {masks_dir}")
        
        if not os.path.exists(images_dir):
            print(f"    WARNING: Images directory not found, skipping batch")
            continue
        if not os.path.exists(masks_dir):
            print(f"    WARNING: Masks directory not found, skipping batch")
            continue
        
        batch_pairs = get_image_mask_pairs(images_dir, masks_dir)
        # Add batch_name to each pair for CLIP lookup
        batch_pairs_with_batch = [
            (img_path, mask_path, img_file, batch_name) 
            for img_path, mask_path, img_file in batch_pairs
        ]
        print(f"    Found {len(batch_pairs)} pairs")
        all_pairs.extend(batch_pairs_with_batch)
    
    print(f"\nTotal pairs from all batches: {len(all_pairs)}")
    
    if len(all_pairs) == 0:
        print("\nERROR: No image-mask pairs found. Please check the directory paths.")
        return
    
    # Count regions in each image and build data structure
    print("\n" + "-" * 80)
    print("Step 2: Analyzing masks and counting regions...")
    print("-" * 80)
    
    data_by_regions = defaultdict(list)
    data_by_strata = defaultdict(list)  # For stratified split (regions + optional vehicle type)
    
    for i, (image_path, mask_path, image_file, batch_name) in enumerate(all_pairs):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(all_pairs)} images...")
        
        num_regions = count_regions_in_mask(mask_path)
        
        # Get vehicle type from CLIP if enabled
        vehicle_type = "unknown"
        if USE_CLIP_CLASSIFICATION and clip_classification:
            vehicle_type = clip_classification.get((batch_name, image_file), "unknown")
        
        sample = {
            'image_file': image_file,
            'image_path': image_path,
            'mask_path': mask_path,
            'batch_name': batch_name,
            'num_regions': num_regions,
            'vehicle_type': vehicle_type
        }
        
        data_by_regions[num_regions].append(sample)
        
        # Create strata key
        if USE_CLIP_CLASSIFICATION:
            strata_key = f"{num_regions}_regions_{vehicle_type}"
        else:
            strata_key = f"{num_regions}_regions"
        
        data_by_strata[strata_key].append(sample)
    
    print(f"Processed all {len(all_pairs)} images")
    
    # Print statistics
    print("\n" + "-" * 80)
    print("Step 3: Dataset statistics")
    print("-" * 80)
    print(f"\nDistribution of {TASK_NAME} regions per image:")
    for num_regions in sorted(data_by_regions.keys()):
        count = len(data_by_regions[num_regions])
        percentage = (count / len(all_pairs)) * 100
        print(f"  {num_regions} region(s): {count:4d} images ({percentage:5.2f}%)")
    
    if USE_CLIP_CLASSIFICATION:
        print(f"\nDistribution by vehicle type:")
        vehicle_counts = defaultdict(int)
        for sample_list in data_by_regions.values():
            for sample in sample_list:
                vehicle_counts[sample['vehicle_type']] += 1
        for vtype, count in sorted(vehicle_counts.items()):
            percentage = (count / len(all_pairs)) * 100
            print(f"  {vtype}: {count:4d} images ({percentage:5.2f}%)")
    
    # Handle empty masks
    excluded_empty = []
    if EXCLUDE_EMPTY_MASKS:
        print("\n" + "-" * 80)
        print(f"Step 4: Excluding images with 0 {TASK_NAME} regions from split")
        print("-" * 80)
        excluded_empty = data_by_regions.pop(0, [])
        
        # Also remove from strata
        strata_keys_to_remove = [k for k in data_by_strata.keys() if k.startswith("0_regions")]
        for key in strata_keys_to_remove:
            del data_by_strata[key]
        
        if excluded_empty:
            print(f"Excluded {len(excluded_empty)} images with 0 regions (will be saved to tmp folder)")
    else:
        print("\n" + "-" * 80)
        print("Step 4: Including all images (including empty masks)")
        print("-" * 80)
    
    # Perform stratified split
    print("\n" + "-" * 80)
    print(f"Step 5: Performing stratified split ({VAL_RATIO*100:.0f}% validation)")
    print("-" * 80)
    
    train_data, val_data = stratified_split(data_by_strata, val_ratio=VAL_RATIO, random_seed=RANDOM_SEED)
    
    # Verify validation set has non-empty masks
    print("\nVerifying validation set masks...")
    val_data_verified = []
    val_rejected = 0
    for sample in val_data:
        if has_non_empty_mask(sample['mask_path']):
            val_data_verified.append(sample)
        else:
            # Move to training (shouldn't happen if EXCLUDE_EMPTY_MASKS is True)
            train_data.append(sample)
            val_rejected += 1
    
    if val_rejected > 0:
        print(f"  Moved {val_rejected} images with empty masks from validation to training")
    
    val_data = val_data_verified
    
    # Calculate totals
    total_after_exclusion = len(train_data) + len(val_data)
    
    if total_after_exclusion == 0:
        print("\nERROR: No images available for split after exclusions.")
        return
    
    print(f"\nSplit results:")
    print(f"  Training set:   {len(train_data):4d} images ({len(train_data)/total_after_exclusion*100:.2f}%)")
    print(f"  Validation set: {len(val_data):4d} images ({len(val_data)/total_after_exclusion*100:.2f}%)")
    if excluded_empty:
        print(f"  Excluded:       {len(excluded_empty):4d} images with 0 regions")
    
    # Show distribution in each split
    print(f"\n  Training set distribution by {TASK_NAME} regions:")
    train_by_regions = defaultdict(int)
    for sample in train_data:
        train_by_regions[sample['num_regions']] += 1
    for num_regions in sorted(train_by_regions.keys()):
        print(f"    {num_regions} region(s): {train_by_regions[num_regions]:4d} images")
    
    print(f"\n  Validation set distribution by {TASK_NAME} regions:")
    val_by_regions = defaultdict(int)
    for sample in val_data:
        val_by_regions[sample['num_regions']] += 1
    for num_regions in sorted(val_by_regions.keys()):
        print(f"    {num_regions} region(s): {val_by_regions[num_regions]:4d} images")
    
    if USE_CLIP_CLASSIFICATION:
        print(f"\n  Validation set distribution by vehicle type:")
        val_by_vehicle = defaultdict(int)
        for sample in val_data:
            val_by_vehicle[sample['vehicle_type']] += 1
        for vtype in sorted(val_by_vehicle.keys()):
            print(f"    {vtype}: {val_by_vehicle[vtype]:4d} images")
    
    # Save special cases and random samples to tmp folder
    print("\n" + "-" * 80)
    print("Step 6: Saving samples to tmp folder")
    print("-" * 80)
    # Re-add excluded images for saving
    if excluded_empty:
        data_by_regions[0] = excluded_empty
    save_special_cases_to_tmp(
        data_by_regions,
        output_dir, 
        special_counts=[0], 
        num_random_samples=NUM_RANDOM_SAMPLES_TMP,
        task_name=TASK_NAME
    )
    
    # Save to JSON
    print("\n" + "-" * 80)
    print("Step 7: Saving split information to JSON")
    print("-" * 80)
    
    split_info = {
        'task': TASK_NAME,
        'mask_pattern': MASK_PATH_PATTERN,
        'batches': BATCH_NAME_LIST,
        'date': date_stamp,
        'total_images': len(all_pairs),
        'excluded_images': len(excluded_empty) if excluded_empty else 0,
        'train_size': len(train_data),
        'val_size': len(val_data),
        'val_ratio': VAL_RATIO,
        'random_seed': RANDOM_SEED,
        'use_clip_classification': USE_CLIP_CLASSIFICATION,
        'exclude_empty_masks': EXCLUDE_EMPTY_MASKS,
        'distribution_by_regions': {
            str(k): len(v) for k, v in sorted(data_by_regions.items())
        },
        'train': train_data,
        'val': val_data,
        'excluded_empty_masks': excluded_empty if excluded_empty else []
    }
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    with open(json_output, 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"Split information saved to: {json_output}")
    
    # Create Roboflow folder structure and copy files
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
    print(f"    train/")
    print(f"      images/    ({len(train_data)} images)")
    print(f"      labels/    ({len(train_data)} masks)")
    print(f"    val/")
    print(f"      images/    ({len(val_data)} images)")
    print(f"      labels/    ({len(val_data)} masks)")
    print(f"    test/")
    print(f"      images/    (empty)")
    print(f"      labels/    (empty)")
    if excluded_empty:
        print(f"    tmp_special_cases_{TASK_NAME}/")
        print(f"      0_regions/ ({len(excluded_empty)} excluded images)")


if __name__ == '__main__':
    main()
