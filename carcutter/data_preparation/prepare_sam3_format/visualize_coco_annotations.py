"""
Visualize COCO annotations to verify they were created correctly.

This script loads COCO annotations and displays images with their
polygon annotations overlaid.
"""

import json
import argparse
from pathlib import Path
import cv2
import numpy as np
from typing import List, Tuple
import random
import os


def load_coco_annotations(json_path: Path) -> dict:
    """Load COCO annotations from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def draw_polygon(image: np.ndarray, polygon: List[float], color: Tuple[int, int, int], thickness: int = 2):
    """Draw a polygon on the image."""
    points = np.array(polygon).reshape(-1, 2).astype(np.int32)
    cv2.polylines(image, [points], isClosed=True, color=color, thickness=thickness)
    
    # Fill with semi-transparent color
    overlay = image.copy()
    cv2.fillPoly(overlay, [points], color)
    cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)


def draw_bbox(image: np.ndarray, bbox: List[float], color: Tuple[int, int, int], thickness: int = 2):
    """Draw a bounding box on the image."""
    x, y, w, h = [int(v) for v in bbox]
    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)


def visualize_sample(
    image_path: Path,
    annotations: List[dict],
    categories: dict,
    show_bbox: bool = True,
    show_polygon: bool = True
) -> np.ndarray:
    """
    Visualize a single image with its annotations.
    
    Args:
        image_path: Path to the image
        annotations: List of annotation dictionaries for this image
        categories: Dictionary mapping category_id to category info
        show_bbox: Whether to show bounding boxes
        show_polygon: Whether to show polygon segmentations
        
    Returns:
        Annotated image as numpy array
    """
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    
    # Generate colors for each instance
    colors = [
        (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        for _ in range(len(annotations))
    ]
    
    # Draw annotations
    for idx, ann in enumerate(annotations):
        color = colors[idx]
        category_id = ann['category_id']
        category_name = categories[category_id]['name']
        
        # Draw polygon
        if show_polygon and 'segmentation' in ann:
            for polygon in ann['segmentation']:
                draw_polygon(image, polygon, color)
        
        # Draw bounding box
        if show_bbox and 'bbox' in ann:
            draw_bbox(image, ann['bbox'], color, thickness=2)
        
        # Add label
        if 'bbox' in ann:
            x, y, w, h = ann['bbox']
            label = f"{category_name} #{idx+1}"
            
            # Add background for text
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                image,
                (int(x), int(y) - text_height - 5),
                (int(x) + text_width, int(y)),
                color,
                -1
            )
            
            # Add text
            cv2.putText(
                image,
                label,
                (int(x), int(y) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
    
    return image


def main():
    parser = argparse.ArgumentParser(
        description="Visualize COCO annotations for SAM3 dataset"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="workspace/data/mirrors/sam3_format",
        help="Root directory containing annotations"
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=None,
        help="Override images directory (default: read from annotation file names or use train/val subdirs)"
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=['train', 'val'],
        default='train',
        help="Which split to visualize (train or val)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of random samples to visualize"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save visualizations (default: display only)"
    )
    parser.add_argument(
        "--no-bbox",
        action="store_true",
        help="Don't show bounding boxes"
    )
    parser.add_argument(
        "--no-polygon",
        action="store_true",
        help="Don't show polygon segmentations"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    dataset_dir = Path(args.dataset_dir)
    annotations_path = dataset_dir / "annotations" / f"instances_{args.split}.json"
    
    # Determine images directory
    if args.images_dir:
        images_dir = Path(args.images_dir)
    else:
        # Try standard location first
        images_dir = dataset_dir / args.split
        if not images_dir.exists():
            # Fall back to original structure
            if args.split == 'train':
                images_dir = dataset_dir.parent / "training" / "train" / "images"
            else:
                images_dir = dataset_dir.parent / "valid" / "images"
    
    # Validate paths
    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotations file not found: {annotations_path}")
    if not images_dir.exists():
        print(f"Warning: Images directory not found at: {images_dir}")
        print("Will try to use absolute paths from annotation file...")
        images_dir = None
    
    # Load annotations
    print(f"Loading annotations from: {annotations_path}")
    coco_data = load_coco_annotations(annotations_path)
    
    # Create category lookup
    categories = {cat['id']: cat for cat in coco_data['categories']}
    
    # Create image_id to annotations mapping
    image_annotations = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in image_annotations:
            image_annotations[img_id] = []
        image_annotations[img_id].append(ann)
    
    # Create image_id to image info mapping
    images = {img['id']: img for img in coco_data['images']}
    
    print(f"\nDataset Statistics:")
    print(f"  Total images: {len(images)}")
    print(f"  Total annotations: {len(coco_data['annotations'])}")
    print(f"  Categories: {[cat['name'] for cat in coco_data['categories']]}")
    print(f"  Images with annotations: {len(image_annotations)}")
    
    # Select random samples
    sample_ids = random.sample(
        list(image_annotations.keys()),
        min(args.num_samples, len(image_annotations))
    )
    
    # Setup output directory if specified
    output_dir = None
    force_save = False
    
    # Check if display is available
    display_available = os.environ.get('DISPLAY', '') != ''
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        force_save = True
        print(f"\nSaving visualizations to: {output_dir}")
    elif not display_available:
        # No display available, auto-save to default location
        output_dir = dataset_dir / "visualizations" / args.split
        output_dir.mkdir(parents=True, exist_ok=True)
        force_save = True
        print(f"\n⚠️  No display detected. Auto-saving visualizations to: {output_dir}")
    
    print(f"\nVisualizing {len(sample_ids)} samples...")
    if not force_save:
        print("Press any key to continue to next image, or 'q' to quit")
    
    # Visualize samples
    for idx, img_id in enumerate(sample_ids, 1):
        image_info = images[img_id]
        annotations = image_annotations[img_id]
        
        # Determine image path
        if images_dir:
            image_path = images_dir / image_info['file_name']
        else:
            # Try to use file_name as absolute path
            image_path = Path(image_info['file_name'])
            if not image_path.exists():
                # Try relative to annotation file
                image_path = annotations_path.parent.parent / image_info['file_name']
        
        print(f"\n[{idx}/{len(sample_ids)}] {image_info['file_name']}")
        print(f"  Image ID: {img_id}")
        print(f"  Dimensions: {image_info['width']}x{image_info['height']}")
        print(f"  Instances: {len(annotations)}")
        
        try:
            # Visualize
            annotated_image = visualize_sample(
                image_path,
                annotations,
                categories,
                show_bbox=not args.no_bbox,
                show_polygon=not args.no_polygon
            )
            
            # Save if output directory specified or no display available
            if output_dir:
                output_path = output_dir / f"vis_{idx:03d}_{image_info['file_name']}"
                cv2.imwrite(str(output_path), annotated_image)
                if force_save:
                    print(f"  ✓ Saved to: {output_path.name}")
            
            # Display only if display is available and not forcing save
            if display_available and not force_save:
                window_name = f"Sample {idx}/{len(sample_ids)} - {image_info['file_name']}"
                cv2.imshow(window_name, annotated_image)
                
                key = cv2.waitKey(0)
                cv2.destroyAllWindows()
                
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    if force_save:
        print(f"\n✓ Visualization complete! All images saved to: {output_dir}")
    else:
        print("\nVisualization complete!")


if __name__ == "__main__":
    main()
