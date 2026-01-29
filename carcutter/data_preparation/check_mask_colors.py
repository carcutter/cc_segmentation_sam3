#!/usr/bin/env python3
"""
Check unique RGB colors in mask images.
Randomly samples X images from a folder and prints unique RGB values for each.
"""

import os
import cv2
import numpy as np
import random
from pathlib import Path


# Color name mapping for common mask colors
COLOR_NAMES = {
    (0, 0, 0): "Black (Background)",
    (255, 255, 255): "White (Foreground/Object)",
    (255, 0, 0): "Red (Car/Vehicle)",
    (0, 255, 0): "Green (Mirror)",
    (0, 0, 255): "Blue",
    (128, 128, 128): "Gray",
    (200, 0, 255): "Purple/Magenta",
    (255, 255, 0): "Yellow",
    (0, 255, 255): "Cyan",
    (255, 0, 255): "Magenta",
    (128, 0, 0): "Dark Red",
    (0, 128, 0): "Dark Green",
    (0, 0, 128): "Dark Blue",
}


def get_color_name(rgb_tuple):
    """
    Get the name of a color from its RGB tuple.
    Returns 'Unknown' if not in the mapping.
    """
    return COLOR_NAMES.get(rgb_tuple, "Unknown")


def get_unique_colors(image_path):
    """
    Get all unique RGB colors in an image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        List of unique RGB tuples
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Reshape to (N, 3) where N is total pixels
    pixels = img_rgb.reshape(-1, 3)
    
    # Get unique colors
    unique_colors = np.unique(pixels, axis=0)
    
    return [tuple(color) for color in unique_colors]


def check_mask_colors(folder_path, num_samples=5):
    """
    Check unique RGB colors in random sample of mask images.
    
    Args:
        folder_path: Path to folder containing mask images
        num_samples: Number of random images to sample
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Error: Folder does not exist: {folder_path}")
        return
    
    # Get all image files
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(folder.glob(f'*{ext}'))
        image_files.extend(folder.glob(f'*{ext.upper()}'))
    
    image_files = sorted(set(image_files))
    
    if len(image_files) == 0:
        print(f"No image files found in {folder_path}")
        return
    
    print(f"Found {len(image_files)} images in folder")
    print(f"Sampling {min(num_samples, len(image_files))} random images\n")
    print("=" * 60)
    
    # Sample random images
    sample_size = min(num_samples, len(image_files))
    sampled_files = random.sample(image_files, sample_size)
    
    # Check each sampled image
    for i, img_path in enumerate(sampled_files, 1):
        print(f"\n[{i}/{sample_size}] {img_path.name}")
        print("-" * 40)
        
        unique_colors = get_unique_colors(img_path)
        
        if unique_colors is None:
            print("  Error: Could not read image")
            continue
        
        print(f"  Found {len(unique_colors)} unique RGB colors:")
        for color in sorted(unique_colors):
            r, g, b = color
            color_name = get_color_name(color)
            print(f"    RGB({r:3d}, {g:3d}, {b:3d}) -> {color_name}")
    
    print("\n" + "=" * 60)
    print("Done!")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Check unique RGB colors in mask images')
    parser.add_argument('folder', type=str, help='Path to folder containing mask images')
    parser.add_argument('-n', '--num_samples', type=int, default=5, 
                        help='Number of random images to sample (default: 5)')
    parser.add_argument('-s', '--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
    
    check_mask_colors(args.folder, args.num_samples)
