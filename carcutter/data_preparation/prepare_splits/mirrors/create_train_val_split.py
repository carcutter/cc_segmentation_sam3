#!/usr/bin/env python3
"""
Script to create train/val split for mirrors dataset with stratification by number of mirrors.
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

def count_mirrors_in_mask(mask_path, min_area=100):
    """
    Count the number of distinct mirrors in a binary mask using connected components.
    Filters out small noise components.
    
    Args:
        mask_path: Path to the mask image
        min_area: Minimum area in pixels for a component to be counted as a mirror
        
    Returns:
        Number of distinct mirror instances (connected components)
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
        num_mirrors = 0
        for i in range(1, num_labels):  # Start from 1 to skip background
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                num_mirrors += 1
        
        return num_mirrors
    except Exception as e:
        print(f"Error processing {mask_path}: {e}")
        return 0


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
            print(f"Warning: No mask found for {image_file}")
    
    return pairs


def stratified_split(data_by_mirrors, val_ratio=0.15, random_seed=42):
    """
    Perform stratified split based on number of mirrors.
    
    Args:
        data_by_mirrors: Dictionary mapping num_mirrors -> list of samples
        val_ratio: Ratio of samples to use for validation
        random_seed: Random seed for reproducibility
        
    Returns:
        train_data, val_data (lists of samples)
    """
    np.random.seed(random_seed)
    
    train_data = []
    val_data = []
    
    for num_mirrors, samples in data_by_mirrors.items():
        # If only 1 image in this group, keep it in training
        if len(samples) == 1:
            print(f"  Group with {num_mirrors} mirror(s) has only 1 image, keeping in training")
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


def save_special_cases_to_tmp(data_by_mirrors, output_dir, special_counts=[0], num_random_samples=5):
    """
    Save images with specific mirror counts to a tmp folder.
    Also saves random samples from all groups for verification.
    
    Args:
        data_by_mirrors: Dictionary mapping num_mirrors -> list of samples
        output_dir: Base output directory
        special_counts: List of mirror counts to save completely (e.g., [0] for exclusions)
        num_random_samples: Number of random samples to save from other groups
    """
    tmp_dir = os.path.join(output_dir, 'tmp_special_cases')
    os.makedirs(tmp_dir, exist_ok=True)
    
    print(f"\nSaving samples to: {tmp_dir}")
    
    for num_mirrors in sorted(data_by_mirrors.keys()):
        samples = data_by_mirrors[num_mirrors]
        subdir = os.path.join(tmp_dir, f'{num_mirrors}_mirrors')
        os.makedirs(os.path.join(subdir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(subdir, 'masks'), exist_ok=True)
        
        # Determine which samples to save
        if num_mirrors in special_counts:
            # Save all samples for special counts
            samples_to_save = samples
            print(f"  Saving all {len(samples)} images with {num_mirrors} mirrors...")
        else:
            # Save random samples for other groups
            n_samples = min(num_random_samples, len(samples))
            np.random.seed(42)
            samples_to_save = np.random.choice(samples, n_samples, replace=False).tolist()
            print(f"  Saving {n_samples} random samples with {num_mirrors} mirrors (out of {len(samples)})...")
        
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
    # Paths
    workspace_root = Path(__file__).parent.parent.parent.parent.parent
    
    # Define batch directories
    root_folder = '/home/raul/workspace/data'
    batch_name_list = ["mirrors","car_segmentation/kw2551_car_segmentation", "car_segmentation/kw2552_car_interior_segmentation", "car_segmentation/kw2553_car_segmentation"]
    
    # Generate date stamp for output
    date_stamp = datetime.now().strftime("%Y%m%d")
    
    # Output directories with date stamp
    output_dir = f'/home/raul/workspace/data/training/mirrors/{date_stamp}'
    json_output = f'{output_dir}/train_val_split_{date_stamp}.json'
    
    print("=" * 80)
    print("MIRRORS DATASET TRAIN/VAL SPLIT")
    print("=" * 80)
    print(f"\nBatches to process: {len(batch_name_list)}")
    for batch_name in batch_name_list:
        print(f"  - {batch_name}")
    print(f"\nOutput directory: {output_dir}")
    print(f"JSON output: {json_output}")
    
    # Get all image-mask pairs from all batches
    print("\n" + "-" * 80)
    print("Step 1: Finding image-mask pairs from all batches...")
    print("-" * 80)
    
    all_pairs = []
    for batch_name in batch_name_list:
        folder_path = f"{root_folder}/{batch_name}"
        images_dir = f"{folder_path}/raw"
        masks_dir = f"{folder_path}/preprocessed_mask/mask"
        
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
        print(f"    Found {len(batch_pairs)} pairs")
        all_pairs.extend(batch_pairs)
    
    print(f"\nTotal pairs from all batches: {len(all_pairs)}")
    
    # Count mirrors in each image
    print("\n" + "-" * 80)
    print("Step 2: Counting mirrors in each image...")
    print("-" * 80)
    data_by_mirrors = defaultdict(list)
    
    for i, (image_path, mask_path, image_file) in enumerate(all_pairs):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(all_pairs)} images...")
        
        num_mirrors = count_mirrors_in_mask(mask_path)
        
        sample = {
            'image_file': image_file,
            'image_path': image_path,
            'mask_path': mask_path,
            'num_mirrors': num_mirrors
        }
        
        data_by_mirrors[num_mirrors].append(sample)
    
    print(f"Processed all {len(all_pairs)} images")
    
    # Print statistics
    print("\n" + "-" * 80)
    print("Step 3: Dataset statistics")
    print("-" * 80)
    print(f"\nDistribution of mirrors per image:")
    for num_mirrors in sorted(data_by_mirrors.keys()):
        count = len(data_by_mirrors[num_mirrors])
        percentage = (count / len(all_pairs)) * 100
        print(f"  {num_mirrors} mirror(s): {count:4d} images ({percentage:5.2f}%)")
    
    # Exclude images with 0 mirrors from training/validation split
    print("\n" + "-" * 80)
    print("Step 4: Excluding images with 0 mirrors from split")
    print("-" * 80)
    excluded_zero_mirrors = data_by_mirrors.pop(0, [])
    if excluded_zero_mirrors:
        print(f"Excluded {len(excluded_zero_mirrors)} images with 0 mirrors (will be saved to tmp folder)")
    
    # Perform stratified split (excluding 0 mirror images)
    print("\n" + "-" * 80)
    print("Step 5: Performing stratified split (15% validation)")
    print("-" * 80)
    train_data, val_data = stratified_split(data_by_mirrors, val_ratio=0.15)
    
    # Calculate total after exclusion
    total_after_exclusion = len(train_data) + len(val_data)
    
    if total_after_exclusion == 0:
        print("\nERROR: No images found in any batch. Please check the directory paths.")
        return
    
    print(f"\nSplit results:")
    print(f"  Training set:   {len(train_data):4d} images ({len(train_data)/total_after_exclusion*100:.2f}%)")
    print(f"  Validation set: {len(val_data):4d} images ({len(val_data)/total_after_exclusion*100:.2f}%)")
    if excluded_zero_mirrors:
        print(f"  Excluded:       {len(excluded_zero_mirrors):4d} images with 0 mirrors")
    
    # Show distribution in each split
    print("\n  Training set distribution:")
    train_by_mirrors = defaultdict(int)
    for sample in train_data:
        train_by_mirrors[sample['num_mirrors']] += 1
    for num_mirrors in sorted(train_by_mirrors.keys()):
        print(f"    {num_mirrors} mirror(s): {train_by_mirrors[num_mirrors]:4d} images")
    
    print("\n  Validation set distribution:")
    val_by_mirrors = defaultdict(int)
    for sample in val_data:
        val_by_mirrors[sample['num_mirrors']] += 1
    for num_mirrors in sorted(val_by_mirrors.keys()):
        print(f"    {num_mirrors} mirror(s): {val_by_mirrors[num_mirrors]:4d} images")
    
    # Save special cases and random samples to tmp folder
    print("\n" + "-" * 80)
    print("Step 6: Saving samples to tmp folder")
    print("-" * 80)
    # Re-add excluded images for saving
    if excluded_zero_mirrors:
        data_by_mirrors[0] = excluded_zero_mirrors
    save_special_cases_to_tmp(data_by_mirrors,
                              output_dir, 
                              special_counts=[0], num_random_samples=5)
    
    # Save to JSON
    print("\n" + "-" * 80)
    print("Step 7: Saving split information to JSON")
    print("-" * 80)
    
    split_info = {
        'batches': batch_name_list,
        'date': date_stamp,
        'total_images': len(all_pairs),
        'excluded_images': len(excluded_zero_mirrors) if excluded_zero_mirrors else 0,
        'train_size': len(train_data),
        'val_size': len(val_data),
        'val_ratio': 0.15,
        'random_seed': 42,
        'distribution': {
            str(k): len(v) for k, v in sorted(data_by_mirrors.items())
        },
        'train': train_data,
        'val': val_data,
        'excluded_0_mirrors': excluded_zero_mirrors if excluded_zero_mirrors else []
    }
    
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


if __name__ == '__main__':
    main()
