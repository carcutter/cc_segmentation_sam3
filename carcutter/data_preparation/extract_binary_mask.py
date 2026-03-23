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


def extract_binary_mask_all_except_background(colored_mask, background_color=(0, 0, 0), tolerance=10):
    """
    Extract binary mask by merging ALL colors except background.
    
    Args:
        colored_mask: BGR image with colored segmentation (each color = category)
        background_color: RGB tuple for background color to exclude (default black: (0, 0, 0))
        tolerance: Color matching tolerance (default 10)
    
    Returns:
        Binary mask where all non-background pixels are white (255)
    """
    # Convert BGR to RGB for easier color specification
    colored_mask_rgb = cv2.cvtColor(colored_mask, cv2.COLOR_BGR2RGB)
    
    # Check if pixel values are within tolerance of background color
    r, g, b = background_color
    r_match = np.abs(colored_mask_rgb[:, :, 0].astype(int) - r) <= tolerance
    g_match = np.abs(colored_mask_rgb[:, :, 1].astype(int) - g) <= tolerance
    b_match = np.abs(colored_mask_rgb[:, :, 2].astype(int) - b) <= tolerance
    
    # Pixels matching background
    is_background = (r_match & g_match & b_match)
    
    # Binary mask is everything that is NOT background
    binary_mask = (~is_background).astype(np.uint8) * 255
    
    return binary_mask


def extract_airholes_from_union(colored_mask, background_color=(0, 0, 0), tolerance=10):
    """
    Extract airholes as background pixels enclosed by any foreground class.

    Args:
        colored_mask: BGR image with colored segmentation (each color = category)
        background_color: RGB tuple for background color to exclude (default black: (0, 0, 0))
        tolerance: Color matching tolerance (default 10)

    Returns:
        Binary mask where enclosed background pixels are white (255)
    """
    # Foreground: any non-background pixel
    foreground_mask = extract_binary_mask_all_except_background(
        colored_mask,
        background_color=background_color,
        tolerance=tolerance
    )

    # Background mask (255 where background)
    background_mask = (foreground_mask == 0).astype(np.uint8) * 255

    # Pad with a 1-pixel background border so (0, 0) is guaranteed background
    padded = cv2.copyMakeBorder(background_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)

    # Flood fill external background from top-left corner
    floodfilled = padded.copy()
    h, w = floodfilled.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(floodfilled, mask, (0, 0), 128)

    # External background is marked as 128
    external_background = (floodfilled == 128).astype(np.uint8) * 255

    # Remove padding
    external_background = external_background[1:-1, 1:-1]

    # Holes: background not connected to border
    airholes = (background_mask == 255) & (external_background == 0)

    return airholes.astype(np.uint8) * 255


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


def fill_holes_in_binary_mask(mask):
    """
    Fill enclosed holes in a binary mask (foreground=255, background=0).

    Args:
        mask: Binary mask (0 or 255)

    Returns:
        Binary mask with internal holes filled (255)
    """
    mask_uint8 = (mask > 0).astype(np.uint8) * 255

    # Background mask (255 where background)
    background_mask = (mask_uint8 == 0).astype(np.uint8) * 255

    # Pad so the flood fill seed is guaranteed background
    padded = cv2.copyMakeBorder(background_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)

    # Flood fill external background from top-left corner
    floodfilled = padded.copy()
    h, w = floodfilled.shape[:2]
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(floodfilled, ff_mask, (0, 0), 128)

    # External background is marked as 128
    external_background = (floodfilled == 128).astype(np.uint8) * 255

    # Remove padding
    external_background = external_background[1:-1, 1:-1]

    # Holes: background not connected to border
    holes = (background_mask == 255) & (external_background == 0)

    # Fill holes
    filled = mask_uint8.copy()
    filled[holes] = 255

    return filled


def process_colored_masks(input_dir, output_dir, category_ids=None, generate_contours=False,
                          erosion_kernel_size=5, erosion_iterations=2,
                          join_all_except_background=False, background_color=(0, 0, 0),
                          extract_airholes_enclosed_background=False, tolerance=10,
                          fill_holes_in_join=False):
    """
    Process all colored masks and extract binary masks for specified categories.
    
    Args:
        input_dir: Directory containing colored segmentation masks
        output_dir: Directory to save binary masks
        category_ids: List of RGB tuples representing categories to extract
                     e.g., [(0, 255, 0)] for green only, [(255, 0, 0), (0, 255, 0)] for red+green
                     Ignored if join_all_except_background is True
        generate_contours: If True, also generate contour masks
        erosion_kernel_size: Kernel size for contour extraction
        erosion_iterations: Iterations for contour extraction
        join_all_except_background: If True, merge ALL colors except background into binary mask
        background_color: RGB tuple for background color (default black: (0, 0, 0))
        extract_airholes_enclosed_background: If True, extract enclosed background pixels as airholes
        tolerance: Color matching tolerance (default 10)
        fill_holes_in_join: If True, fill enclosed holes when join_all_except_background is True
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
    if extract_airholes_enclosed_background:
        print("Mode: Extract airholes from enclosed background")
        print(f"Background color: {background_color}, tolerance: {tolerance}")
    elif join_all_except_background:
        print(f"Mode: Join ALL colors except background {background_color}")
        if fill_holes_in_join:
            print("Hole fill: enabled")
    else:
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
        
        # Extract binary mask based on mode
        if extract_airholes_enclosed_background:
            binary_mask = extract_airholes_from_union(
                colored_mask,
                background_color=background_color,
                tolerance=tolerance
            )
        elif join_all_except_background:
            binary_mask = extract_binary_mask_all_except_background(
                colored_mask,
                background_color=background_color,
                tolerance=tolerance
            )
            if fill_holes_in_join:
                binary_mask = fill_holes_in_binary_mask(binary_mask)
        else:
            binary_mask = extract_binary_mask_by_ids(
                colored_mask,
                category_ids,
                tolerance=tolerance
            )
        
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
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract binary masks from colored segmentation masks."
    )
    parser.add_argument("--input-dir", required=True, help="Path to colored mask images")
    parser.add_argument("--output-dir", required=True, help="Path to save binary masks")
    parser.add_argument(
        "--mode", required=True, choices=["all_except_background", "by_color_ids"],
        help="Extraction mode: 'all_except_background' merges all non-background colors; "
             "'by_color_ids' extracts specific colors (white by default)"
    )
    parser.add_argument(
        "--background-color", default="0,0,0",
        help="RGB background color as comma-separated ints (default: 0,0,0)"
    )
    parser.add_argument(
        "--tolerance", type=int, default=10,
        help="Color matching tolerance (default: 10)"
    )
    parser.add_argument(
        "--fill-holes", action="store_true",
        help="Fill enclosed holes when using all_except_background mode"
    )
    parser.add_argument(
        "--generate-contours", action="store_true",
        help="Also generate contour masks alongside binary masks"
    )
    args = parser.parse_args()

    bg = tuple(int(x) for x in args.background_color.split(","))
    join_all = args.mode == "all_except_background"
    # Default color IDs for by_color_ids mode: white (common for single-class masks)
    category_ids = [(255, 255, 255)]

    process_colored_masks(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        category_ids=category_ids,
        generate_contours=args.generate_contours,
        erosion_kernel_size=5,
        erosion_iterations=2,
        join_all_except_background=join_all,
        background_color=bg,
        extract_airholes_enclosed_background=False,
        tolerance=args.tolerance,
        fill_holes_in_join=args.fill_holes,
    )


if __name__ == '__main__':
    main()
