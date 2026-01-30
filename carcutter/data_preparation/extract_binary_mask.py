#!/usr/bin/env python3
"""
Extract binary masks from colored segmentation masks.
Supports extracting specific category IDs and merging multiple categories.
Optionally generates contour masks.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


def extract_binary_mask_by_ids(colored_mask, category_ids, tolerance=10):
    """
    Extract binary mask from colored segmentation mask based on category IDs.
    
    Args:
        colored_mask: BGR image with colored segmentation (each color = category)
        category_ids: List of RGB tuples representing categories to extract
                     e.g., [(0, 255, 0)] for green, [(255, 0, 0), (0, 255, 0)] for red+green
        tolerance: Color matching tolerance (default 10)
    
    Returns:
        Binary mask where specified categories are white (255) and rest is black (0)
    """
    # Convert BGR to RGB for easier color specification
    colored_mask_rgb = cv2.cvtColor(colored_mask, cv2.COLOR_BGR2RGB)
    
    # Create empty binary mask
    binary_mask = np.zeros(colored_mask_rgb.shape[:2], dtype=np.uint8)
    
    # Extract each category and merge them
    for category_rgb in category_ids:
        # Create mask for this specific category
        r, g, b = category_rgb
        
        # Check if pixel values are within tolerance of target color
        r_match = np.abs(colored_mask_rgb[:, :, 0].astype(int) - r) <= tolerance
        g_match = np.abs(colored_mask_rgb[:, :, 1].astype(int) - g) <= tolerance
        b_match = np.abs(colored_mask_rgb[:, :, 2].astype(int) - b) <= tolerance
        
        # All channels must match
        category_mask = (r_match & g_match & b_match).astype(np.uint8)
        
        # Merge with binary mask (logical OR)
        binary_mask = np.maximum(binary_mask, category_mask)
    
    # Convert to 0-255 range
    binary_mask = binary_mask * 255
    
    return binary_mask


def extract_contour_mask(mask, erosion_kernel_size=5, erosion_iterations=2):
    """
    Extract contours from a binary mask by erosion only (inner boundary).
    
    Args:
        mask: Binary mask (0 or 255)
        erosion_kernel_size: Size of erosion kernel
        erosion_iterations: Number of erosion iterations
    
    Returns:
        Contour mask (inner boundary only)
    """
    mask_uint8 = mask.astype(np.uint8)
    
    # Erode the mask to create inner boundary
    erosion_kernel = np.ones((erosion_kernel_size, erosion_kernel_size), np.uint8)
    eroded_mask = cv2.erode(mask_uint8, erosion_kernel, iterations=erosion_iterations)
    
    # The contour is the difference between original and eroded mask
    # This only includes pixels that were inside the original mask
    contour_mask = cv2.subtract(mask_uint8, eroded_mask)
    
    return contour_mask


def process_colored_masks(input_dir, output_dir, category_ids, generate_contours=False,
                          erosion_kernel_size=5, erosion_iterations=2):
    """
    Process all colored masks and extract binary masks for specified categories.
    
    Args:
        input_dir: Directory containing colored segmentation masks
        output_dir: Directory to save binary masks
        category_ids: List of RGB tuples representing categories to extract
                     e.g., [(0, 255, 0)] for green only, [(255, 0, 0), (0, 255, 0)] for red+green
        generate_contours: If True, also generate contour masks
        erosion_kernel_size: Kernel size for contour extraction
        erosion_iterations: Iterations for contour extraction
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create contour output directory if needed
    if generate_contours:
        # Contour folder is one level back from output_dir, in a folder called "contour_masks"
        parent_dir = os.path.dirname(output_dir)
        contour_dir = os.path.join(parent_dir, "contour_masks")
        os.makedirs(contour_dir, exist_ok=True)
        print(f"Contour masks will be saved to: {contour_dir}")
    
    # Get all image files
    input_path = Path(input_dir)
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    mask_files = []
    for ext in image_extensions:
        mask_files.extend(input_path.glob(f'*{ext}'))
    
    mask_files = sorted(mask_files)
    
    if len(mask_files) == 0:
        print(f"No mask files found in {input_dir}")
        return
    
    print(f"Found {len(mask_files)} mask files")
    print(f"Processing masks from: {input_dir}")
    print(f"Saving binary masks to: {output_dir}")
    print(f"Categories to extract (RGB): {category_ids}")
    if generate_contours:
        print(f"Contour params: erosion_kernel={erosion_kernel_size}, erosion_iters={erosion_iterations}")
    
    # Statistics
    stats = {
        'total': len(mask_files),
        'with_objects': 0,
        'without_objects': 0
    }
    
    # Process each mask
    for mask_file in tqdm(mask_files, desc="Extracting binary masks"):
        # Load colored mask
        colored_mask = cv2.imread(str(mask_file))
        
        if colored_mask is None:
            print(f"Warning: Could not read {mask_file}, skipping...")
            continue
        
        # Extract binary mask for specified categories
        binary_mask = extract_binary_mask_by_ids(colored_mask, category_ids)
        
        # Update statistics
        if np.max(binary_mask) > 0:
            stats['with_objects'] += 1
        else:
            stats['without_objects'] += 1
        
        # Save binary mask
        output_path = os.path.join(output_dir, mask_file.name)
        cv2.imwrite(output_path, binary_mask)
        
        # Generate and save contour mask if requested
        if generate_contours:
            if np.max(binary_mask) == 0:
                # Empty mask - save black contour mask
                contour_mask = np.zeros_like(binary_mask)
            else:
                # Extract contour from binary mask
                contour_mask = extract_contour_mask(
                    binary_mask,
                    erosion_kernel_size=erosion_kernel_size,
                    erosion_iterations=erosion_iterations
                )
            
            # Save contour mask
            contour_path = os.path.join(contour_dir, mask_file.name)
            cv2.imwrite(contour_path, contour_mask)
    
            # Save contour mask
            contour_path = os.path.join(contour_dir, mask_file.name)
            cv2.imwrite(contour_path, contour_mask)
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"{'='*60}")
    print(f"Total masks processed: {stats['total']}")
    print(f"Masks with objects: {stats['with_objects']}")
    print(f"Masks without objects: {stats['without_objects']}")
    if generate_contours:
        print(f"Contour masks generated: {stats['total']}")
    print(f"{'='*60}")


def main():
    """
    Main function to extract binary masks from colored segmentation masks.
    
    Configure:
    - input_dir: Path to colored mask images
    - output_dir: Path to save binary masks
    - category_ids: List of RGB tuples for categories to extract and merge
      Examples:
        - [(0, 255, 0)] -> Extract only green pixels (mirrors)
        - [(255, 0, 0)] -> Extract only red pixels (cars)
        - [(255, 0, 0), (0, 255, 0)] -> Extract both red and green (cars + mirrors)
        - [(0, 0, 0)] -> Extract black pixels (background)
    - generate_contours: Set to True to also generate contour masks
    - erosion_kernel_size: Kernel size for contour extraction (default: 5)
    - erosion_iterations: Number of erosion iterations for contour (default: 2)
    """
    # ========================= CONFIGURATION =========================
    
    # Input/output paths
    batch_name_list = ["kw2551_car_segmentation", "kw2552_car_interior_segmentation", "kw2553_car_segmentation", "kw2602_car_segmentation"]
    task_name =  "outline" # "holes"
    for batch_name in batch_name_list:
        folder_path = f"/home/raul/workspace/data/car_segmentation/{batch_name}"
        input_dir = f"{folder_path}/masks"
        output_dir = f"{folder_path}/preprocessed_mask/{task_name}/mask"
        
        # Category IDs to extract (RGB format)
        # Example: Extract green (mirror) pixels only
        # category_ids =  [(0, 255, 0)]  # Green color in RGB
        #category_ids = [(  0,   0, 255)] #-> Blue
        category_ids = [ 
            (255,   0,   0), #-> Red (Car/Vehicle)
            (0,   255,   0),  #-> Green (Mirror)
            (128, 128, 128),  # -> Gray
            (200,   0, 255),  # -> Purple/Magenta
            (255,   0,   0),  # -> Red (Car/Vehicle)
            (255, 165,   0),  # -> Orange
            (  0,   0, 255) #-> Blue
        ]

        # To extract multiple categories (merge them into one binary mask):
        # category_ids = [(255, 0, 0), (0, 255, 0)]  # Red + Green
        
        # Contour generation settings
        generate_contours = True  # Set to True to generate contour masks
        erosion_kernel_size = 5    # Kernel size for contour extraction
        erosion_iterations = 2     # Iterations for contour extraction
        
        # ================================================================
        
        # Process masks
        process_colored_masks(
            input_dir=input_dir,
            output_dir=output_dir,
            category_ids=category_ids,
            generate_contours=generate_contours,
            erosion_kernel_size=erosion_kernel_size,
            erosion_iterations=erosion_iterations
        )


if __name__ == '__main__':
    main()
