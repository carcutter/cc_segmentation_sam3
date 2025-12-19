#!/usr/bin/env python3
"""
Script to create train/val split for mirrors dataset with stratification by number of mirrors.
"""

import os
import json
import shutil
from pathlib import Path
from collections import defaultdict
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
        List of tuples (image_path, mask_path, image_basename)
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


def save_special_cases_to_tmp(data_by_mirrors, images_dir, masks_dir, output_dir, special_counts=[0], num_random_samples=5):
    """
    Save images with specific mirror counts to a tmp folder.
    Also saves random samples from all groups for verification.
    
    Args:
        data_by_mirrors: Dictionary mapping num_mirrors -> list of samples
        images_dir: Source images directory
        masks_dir: Source masks directory
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
            
            # Source paths
            src_image = os.path.join(images_dir, image_file)
            src_mask = os.path.join(masks_dir, image_basename + '.png')
            
            # Destination paths
            dst_image = os.path.join(subdir, 'images', image_file)
            dst_mask = os.path.join(subdir, 'masks', image_basename + '.png')
            
            shutil.copy2(src_image, dst_image)
            shutil.copy2(src_mask, dst_mask)


def copy_files_to_roboflow_structure(train_data, val_data, output_dir, images_dir, masks_dir):
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
    splits = ['train', 'val, 'test']
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
        
        # Source paths
        src_image = os.path.join(images_dir, image_file)
        src_mask = os.path.join(masks_dir, image_basename + '.png')
        
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
        
        # Source paths
        src_image = os.path.join(images_dir, image_file)
        src_mask = os.path.join(masks_dir, image_basename + '.png')
        
        # Destination paths
        dst_image = os.path.join(output_dir, 'val', 'images', image_file)
        dst_mask = os.path.join(output_dir, 'val', 'labels', image_basename + '.png')
        
        shutil.copy2(src_image, dst_image)
        shutil.copy2(src_mask, dst_mask)
    
    print(f"\nTest folder created (empty) at: {os.path.join(output_dir, 'test')}")


def main():
    # Paths
    workspace_root = Path(__file__).parent.parent.parent.parent.parent
    images_dir = workspace_root / 'data' / 'mirrors' / 'raw'
    masks_dir = workspace_root / 'data' / 'mirrors' / 'preprocessed_mask' / 'mask'
    output_dir = workspace_root / 'data' / 'mirrors' / 'training'
    json_output = workspace_root / 'data' / 'mirrors' / 'train_val_split.json'
    
    # Convert to strings
    images_dir = str(images_dir)
    masks_dir = str(masks_dir)
    output_dir = str(output_dir)
    json_output = str(json_output)
    
    print("=" * 80)
    print("MIRRORS DATASET TRAIN/VAL SPLIT")
    print("=" * 80)
    print(f"\nImages directory: {images_dir}")
    print(f"Masks directory: {masks_dir}")
    print(f"Output directory: {output_dir}")
    print(f"JSON output: {json_output}")
    
    # Get all image-mask pairs
    print("\n" + "-" * 80)
    print("Step 1: Finding image-mask pairs...")
    print("-" * 80)
    pairs = get_image_mask_pairs(images_dir, masks_dir)
    print(f"Found {len(pairs)} image-mask pairs")
    
    # Count mirrors in each image
    print("\n" + "-" * 80)
    print("Step 2: Counting mirrors in each image...")
    print("-" * 80)
    data_by_mirrors = defaultdict(list)
    
    for i, (image_path, mask_path, image_file) in enumerate(pairs):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(pairs)} images...")
        
        num_mirrors = count_mirrors_in_mask(mask_path)
        
        sample = {
            'image_file': image_file,
            'num_mirrors': num_mirrors
        }
        
        data_by_mirrors[num_mirrors].append(sample)
    
    print(f"Processed all {len(pairs)} images")
    
    # Print statistics
    print("\n" + "-" * 80)
    print("Step 3: Dataset statistics")
    print("-" * 80)
    print(f"\nDistribution of mirrors per image:")
    for num_mirrors in sorted(data_by_mirrors.keys()):
        count = len(data_by_mirrors[num_mirrors])
        percentage = (count / len(pairs)) * 100
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
    save_special_cases_to_tmp(data_by_mirrors, images_dir, masks_dir, 
                              str(workspace_root / 'data' / 'mirrors'), 
                              special_counts=[0], num_random_samples=5)
    
    # Save to JSON
    print("\n" + "-" * 80)
    print("Step 7: Saving split information to JSON")
    print("-" * 80)
    
    split_info = {
        'total_images': len(pairs),
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
    
    copy_files_to_roboflow_structure(train_data, val_data, output_dir, images_dir, masks_dir)
    
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
