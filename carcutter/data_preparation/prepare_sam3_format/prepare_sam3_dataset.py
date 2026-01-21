"""
Convert segmentation dataset with binary masks to COCO format for SAM3 finetuning.

This script works with existing train/val splits:
- train/images -> training images
- train/labels -> training masks
- val/images -> validation images
- val/labels -> validation masks

The script:
1. Reads images and their corresponding binary masks from existing splits
2. Extracts individual object instances from each mask
3. Generates polygon annotations for each instance
4. Creates COCO-format JSON annotations
5. Optionally copies/organizes data in the expected structure for SAM3 training
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
import shutil


def find_matching_mask(image_path: Path, mask_dir: Path) -> Path:
    """Find the mask file matching the image basename."""
    image_stem = image_path.stem
    
    # Try common image extensions for masks
    for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
        mask_path = mask_dir / f"{image_stem}{ext}"
        if mask_path.exists():
            return mask_path
    
    raise FileNotFoundError(f"No matching mask found for {image_path.name}")


def extract_polygons_from_mask(mask: np.ndarray, min_area: int = 100) -> List[List[float]]:
    """
    Extract polygon contours from a binary mask.
    
    Args:
        mask: Binary mask where white (255) represents the object
        min_area: Minimum area threshold to filter out small noise
        
    Returns:
        List of polygons, where each polygon is [x1, y1, x2, y2, ...]
    """
    # Ensure mask is binary
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    # Threshold to ensure binary
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for contour in contours:
        # Filter small contours
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        # Simplify contour to reduce points
        epsilon = 0.001 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Convert to flat list [x1, y1, x2, y2, ...]
        polygon = approx.flatten().tolist()
        
        # Need at least 6 coordinates (3 points) for a valid polygon
        if len(polygon) >= 6:
            polygons.append(polygon)
    
    return polygons


def calculate_bbox_from_polygon(polygon: List[float]) -> List[float]:
    """Calculate bounding box [x, y, width, height] from polygon."""
    xs = polygon[0::2]
    ys = polygon[1::2]
    
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def calculate_area_from_polygon(polygon: List[float]) -> float:
    """Calculate area of a polygon using the Shoelace formula."""
    xs = polygon[0::2]
    ys = polygon[1::2]
    
    n = len(xs)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += xs[i] * ys[j]
        area -= xs[j] * ys[i]
    
    return abs(area) / 2.0


def process_dataset(
    train_images_dir: Path,
    train_labels_dir: Path,
    val_images_dir: Path,
    val_labels_dir: Path,
    output_dir: Path,
    category_names: List[str] = None,
    min_area: int = 100,
    copy_images: bool = False
) -> Dict:
    """
    Process the dataset and create COCO annotations from existing train/val splits.
    
    Args:
        train_images_dir: Directory containing training images
        train_labels_dir: Directory containing training masks
        val_images_dir: Directory containing validation images
        val_labels_dir: Directory containing validation masks
        output_dir: Directory to save processed dataset
        category_names: List of category names (different text prompts for the same object)
        min_area: Minimum area threshold for filtering noise
        copy_images: Whether to copy images to output directory (default: False, uses symlinks or relative paths)
        
    Returns:
        Dictionary with statistics about the processing
    """
    # Default to single 'mirror' category if none specified
    if category_names is None:
        category_names = ["mirror"]
    
    # Create output structure
    output_dir = Path(output_dir)
    annotations_dir = output_dir / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    
    # Optionally create image directories
    if copy_images:
        output_train_dir = output_dir / "train"
        output_val_dir = output_dir / "val"
        output_train_dir.mkdir(parents=True, exist_ok=True)
        output_val_dir.mkdir(parents=True, exist_ok=True)
    else:
        # Use original image directories
        output_train_dir = train_images_dir
        output_val_dir = val_images_dir
    
    # Initialize COCO structure for train and val
    current_date = datetime.now()
    
    # Create multiple categories with different prompts (all refer to same objects)
    categories = [
        {
            "id": idx + 1,
            "name": cat_name,
            "supercategory": "object"
        }
        for idx, cat_name in enumerate(category_names)
    ]
    
    coco_train = {
        "info": {
            "description": f"{category_names[0]} Detection Dataset",
            "version": "1.0",
            "year": current_date.year,
            "contributor": "",
            "date_created": current_date.strftime("%Y-%m-%d")
        },
        "images": [],
        "annotations": [],
        "categories": categories
    }
    
    coco_val = {
        "info": {
            "description": f"{category_names[0]} Detection Dataset",
            "version": "1.0",
            "year": current_date.year,
            "contributor": "",
            "date_created": current_date.strftime("%Y-%m-%d")
        },
        "images": [],
        "annotations": [],
        "categories": categories
    }
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    train_files = [f for f in Path(train_images_dir).iterdir() 
                   if f.suffix.lower() in image_extensions]
    val_files = [f for f in Path(val_images_dir).iterdir() 
                 if f.suffix.lower() in image_extensions]
    
    print(f"Found {len(train_files)} training images")
    print(f"Found {len(val_files)} validation images")
    
    stats = {
        "total_images": len(train_files) + len(val_files),
        "train_images": len(train_files),
        "val_images": len(val_files),
        "train_instances": 0,
        "val_instances": 0,
        "skipped_images": 0
    }
    
    annotation_id = 1
    
    # Process training set
    print("\nProcessing training set...")
    for img_id, image_path in enumerate(tqdm(train_files), start=1):
        try:
            mask_path = find_matching_mask(image_path, train_labels_dir)
            result = process_single_image(
                image_path, mask_path, img_id, annotation_id,
                output_train_dir, min_area, category_ids=list(range(1, len(category_names) + 1)),
                copy_image=copy_images
            )
            
            if result:
                coco_train["images"].append(result["image_info"])
                coco_train["annotations"].extend(result["annotations"])
                annotation_id = result["next_annotation_id"]
                stats["train_instances"] += len(result["annotations"])
            else:
                stats["skipped_images"] += 1
                
        except FileNotFoundError as e:
            print(f"\nWarning: {e}")
            stats["skipped_images"] += 1
            continue
    
    # Process validation set
    print("\nProcessing validation set...")
    for img_id, image_path in enumerate(tqdm(val_files), start=1):
        try:
            mask_path = find_matching_mask(image_path, val_labels_dir)
            result = process_single_image(
                image_path, mask_path, img_id, annotation_id,
                output_val_dir, min_area, category_ids=list(range(1, len(category_names) + 1)),
                copy_image=copy_images
            )
            
            if result:
                coco_val["images"].append(result["image_info"])
                coco_val["annotations"].extend(result["annotations"])
                annotation_id = result["next_annotation_id"]
                stats["val_instances"] += len(result["annotations"])
            else:
                stats["skipped_images"] += 1
                
        except FileNotFoundError as e:
            print(f"\nWarning: {e}")
            stats["skipped_images"] += 1
            continue
    
    # Save annotations
    train_ann_path = annotations_dir / "instances_train.json"
    val_ann_path = annotations_dir / "instances_val.json"
    
    with open(train_ann_path, 'w') as f:
        json.dump(coco_train, f, indent=2)
    
    with open(val_ann_path, 'w') as f:
        json.dump(coco_val, f, indent=2)
    
    print(f"\nSaved annotations to:")
    print(f"  Train: {train_ann_path}")
    print(f"  Val: {val_ann_path}")
    
    return stats


def process_single_image(
    image_path: Path,
    mask_path: Path,
    image_id: int,
    annotation_id: int,
    output_images_dir: Path,
    min_area: int,
    category_ids: List[int],
    copy_image: bool = False
) -> Optional[Dict]:
    """Process a single image-mask pair."""
    # Read image to get dimensions
    image = Image.open(image_path)
    width, height = image.size
    
    # Copy or use original image path
    if copy_image:
        output_image_path = output_images_dir / image_path.name
        image.save(output_image_path)
        file_name = image_path.name
    else:
        # Use relative or absolute path
        file_name = image_path.name
    
    # Read mask
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    
    # Extract polygons for each instance
    polygons = extract_polygons_from_mask(mask, min_area)
    
    if not polygons:
        return None
    
    # Create image info
    image_info = {
        "id": image_id,
        "file_name": file_name,
        "height": height,
        "width": width
    }
    
    # Create annotations for each polygon (each mirror instance)
    # For each instance, create one annotation per category (enables multi-prompt training)
    annotations = []
    for polygon in polygons:
        bbox = calculate_bbox_from_polygon(polygon)
        area = calculate_area_from_polygon(polygon)
        
        # Create one annotation for each category ID (all refer to same object)
        for category_id in category_ids:
            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": [polygon],  # COCO format expects list of polygons
                "area": area,
                "bbox": bbox,  # [x, y, width, height]
                "iscrowd": 0
            }
            
            annotations.append(annotation)
            annotation_id += 1
    
    return {
        "image_info": image_info,
        "annotations": annotations,
        "next_annotation_id": annotation_id
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prepare segmentation dataset for SAM3 finetuning from existing train/val splits"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root directory for the task (e.g., /path/to/data/training/mirrors or /path/to/data/training/mirrors/20260120)"
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Date folder to use (e.g., 20260120). If not specified, uses latest date folder or assumes data-root already includes date."
    )
    parser.add_argument(
        "--task-name",
        type=str,
        required=True,
        help="Name of the task/category (e.g., 'mirror', 'license_plate', 'car')"
    )
    parser.add_argument(
        "--train-images",
        type=str,
        default=None,
        help="Training images directory (default: {data-root}/train/images)"
    )
    parser.add_argument(
        "--train-labels",
        type=str,
        default=None,
        help="Training labels directory (default: {data-root}/train/labels)"
    )
    parser.add_argument(
        "--val-images",
        type=str,
        default=None,
        help="Validation images directory (default: {data-root}/val/images)"
    )
    parser.add_argument(
        "--val-labels",
        type=str,
        default=None,
        help="Validation labels directory (default: {data-root}/val/labels)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for COCO-formatted dataset (default: {data-root}/sam3_format)"
    )
    parser.add_argument(
        "--category-names",
        type=str,
        nargs="+",
        default=None,
        help="List of category names (text prompts) for the object, e.g., 'mirror' 'reflective surface' 'glass mirror' (default: task-name only)"
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=100,
        help="Minimum area threshold to filter small noise"
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images to output directory (default: keep original locations)"
    )
    
    args = parser.parse_args()
    
    # Setup paths with defaults
    data_root = Path(args.data_root)
    
    # Handle date-stamped directories
    # If date is specified, append it to data_root
    if args.date:
        data_root = data_root / args.date
        print(f"Using specified date folder: {args.date}")
    # If data_root doesn't contain train/val, try to auto-detect latest date folder
    elif not (data_root / "train").exists():
        # Look for date-stamped directories (YYYYMMDD format)
        date_folders = [d for d in data_root.iterdir() 
                       if d.is_dir() and d.name.isdigit() and len(d.name) == 8]
        if date_folders:
            # Use the latest date folder
            latest_date = sorted(date_folders)[-1]
            data_root = latest_date
            print(f"Auto-detected latest date folder: {latest_date.name}")
        else:
            print("Warning: No train folder found and no date folders detected.")
            print(f"Assuming data_root is correct: {data_root}")
    
    # Set category names (use task-name if not specified)
    if args.category_names:
        category_names = args.category_names
    else:
        # Default: use task-name as single category
        category_names = [args.task_name]
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = data_root / "sam3_format"
    
    # Setup training paths
    if args.train_images:
        train_images_dir = Path(args.train_images)
    else:
        train_images_dir = data_root / "train" / "images"
    
    if args.train_labels:
        train_labels_dir = Path(args.train_labels)
    else:
        train_labels_dir = data_root / "train" / "labels"
    
    # Setup validation paths
    if args.val_images:
        val_images_dir = Path(args.val_images)
    else:
        val_images_dir = data_root / "val" / "images"
    
    if args.val_labels:
        val_labels_dir = Path(args.val_labels)
    else:
        val_labels_dir = data_root / "val" / "labels"
    
    # Validate input directories
    for dir_path, name in [
        (train_images_dir, "Training images"),
        (train_labels_dir, "Training labels"),
        (val_images_dir, "Validation images"),
        (val_labels_dir, "Validation labels")
    ]:
        if not dir_path.exists():
            raise FileNotFoundError(f"{name} directory not found: {dir_path}")
    
    print("=" * 60)
    print(f"SAM3 Dataset Preparation - Task: {args.task_name}")
    print("=" * 60)
    print(f"Data root: {data_root}")
    print(f"Training images: {train_images_dir}")
    print(f"Training labels: {train_labels_dir}")
    print(f"Validation images: {val_images_dir}")
    print(f"Validation labels: {val_labels_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Category names (prompts): {', '.join(category_names)}")
    print(f"Min area: {args.min_area}")
    print(f"Copy images: {args.copy_images}")
    print("=" * 60)
    
    # Process dataset
    stats = process_dataset(
        train_images_dir=train_images_dir,
        train_labels_dir=train_labels_dir,
        val_images_dir=val_images_dir,
        val_labels_dir=val_labels_dir,
        output_dir=output_dir,
        category_names=category_names,
        min_area=args.min_area,
        copy_images=args.copy_images
    )
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    print(f"Total images processed: {stats['total_images']}")
    print(f"Skipped images: {stats['skipped_images']}")
    print(f"\nTraining set:")
    print(f"  Images: {stats['train_images']}")
    print(f"  Object instances: {stats['train_instances']}")
    print(f"  Avg instances per image: {stats['train_instances']/max(stats['train_images'], 1):.2f}")
    print(f"\nValidation set:")
    print(f"  Images: {stats['val_images']}")
    print(f"  Object instances: {stats['val_instances']}")
    print(f"  Avg instances per image: {stats['val_instances']/max(stats['val_images'], 1):.2f}")
    print("=" * 60)
    
    if args.copy_images:
        print(f"\nDataset structure:")
        print(f"  {output_dir}/")
        print(f"  ├── train/          (training images - copied)")
        print(f"  ├── val/            (validation images - copied)")
        print(f"  └── annotations/")
        print(f"      ├── instances_train.json")
        print(f"      └── instances_val.json")
    else:
        print(f"\nAnnotations saved to:")
        print(f"  {output_dir}/annotations/")
        print(f"  ├── instances_train.json")
        print(f"  └── instances_val.json")
        print(f"\nImages remain at original locations:")
        print(f"  Training: {train_images_dir}")
        print(f"  Validation: {val_images_dir}")
    
    print(f"\nYou can now use this dataset with SAM3 training.")
    print(f"Update your SAM3 config to point to: {output_dir}")


if __name__ == "__main__":
    main()
