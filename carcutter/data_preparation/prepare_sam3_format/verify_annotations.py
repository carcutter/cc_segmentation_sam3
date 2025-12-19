"""
Verify that the generated COCO annotations are valid for SAM3 training.

This script checks:
1. JSON structure (images, annotations, categories)
2. Required fields in each annotation
3. Polygon format validity
4. Image-annotation consistency
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


def verify_coco_annotations(ann_file: Path):
    """Verify COCO annotation file structure and content."""
    print(f"\nVerifying: {ann_file}")
    print("=" * 60)
    
    with open(ann_file, 'r') as f:
        data = json.load(f)
    
    # Check top-level structure
    required_keys = ['images', 'annotations', 'categories']
    missing_keys = [k for k in required_keys if k not in data]
    
    if missing_keys:
        print(f"❌ ERROR: Missing required keys: {missing_keys}")
        return False
    
    print(f"✓ Valid COCO structure")
    
    # Verify categories
    print(f"\n📋 Categories: {len(data['categories'])}")
    for cat in data['categories']:
        required_cat_fields = ['id', 'name']
        if all(f in cat for f in required_cat_fields):
            print(f"  - {cat['id']}: {cat['name']}")
        else:
            print(f"  ❌ Invalid category: {cat}")
            return False
    
    # Verify images
    print(f"\n🖼️  Images: {len(data['images'])}")
    image_ids = set()
    required_img_fields = ['id', 'file_name', 'height', 'width']
    
    for img in data['images']:
        if not all(f in img for f in required_img_fields):
            print(f"  ❌ Invalid image entry: missing fields")
            return False
        image_ids.add(img['id'])
    
    print(f"  ✓ All images have required fields")
    
    # Verify annotations
    print(f"\n📍 Annotations: {len(data['annotations'])}")
    required_ann_fields = ['id', 'image_id', 'category_id', 'bbox', 'area', 'segmentation']
    
    annotations_per_image = defaultdict(int)
    invalid_anns = []
    segmentation_formats = defaultdict(int)
    
    for ann in data['annotations']:
        # Check required fields
        if not all(f in ann for f in required_ann_fields):
            missing = [f for f in required_ann_fields if f not in ann]
            invalid_anns.append(f"Annotation {ann.get('id', '?')}: missing {missing}")
            continue
        
        # Check image_id is valid
        if ann['image_id'] not in image_ids:
            invalid_anns.append(f"Annotation {ann['id']}: invalid image_id {ann['image_id']}")
            continue
        
        # Check bbox format [x, y, width, height]
        if not (isinstance(ann['bbox'], list) and len(ann['bbox']) == 4):
            invalid_anns.append(f"Annotation {ann['id']}: invalid bbox format")
            continue
        
        # Check segmentation format
        seg = ann['segmentation']
        if isinstance(seg, list) and len(seg) > 0:
            if isinstance(seg[0], list):
                # Polygon format [[x1, y1, x2, y2, ...]]
                segmentation_formats['polygon'] += 1
            else:
                invalid_anns.append(f"Annotation {ann['id']}: invalid polygon format")
        elif isinstance(seg, dict) and 'counts' in seg:
            # RLE format
            segmentation_formats['RLE'] += 1
        else:
            invalid_anns.append(f"Annotation {ann['id']}: invalid segmentation format")
        
        annotations_per_image[ann['image_id']] += 1
    
    if invalid_anns:
        print(f"  ❌ Found {len(invalid_anns)} invalid annotations:")
        for msg in invalid_anns[:10]:  # Show first 10
            print(f"    - {msg}")
        if len(invalid_anns) > 10:
            print(f"    ... and {len(invalid_anns) - 10} more")
        return False
    
    print(f"  ✓ All annotations valid")
    print(f"  Segmentation formats:")
    for fmt, count in segmentation_formats.items():
        print(f"    - {fmt}: {count}")
    
    # Statistics
    if annotations_per_image:
        avg_anns = sum(annotations_per_image.values()) / len(annotations_per_image)
        max_anns = max(annotations_per_image.values())
        min_anns = min(annotations_per_image.values())
        
        print(f"\n📊 Statistics:")
        print(f"  Average annotations per image: {avg_anns:.2f}")
        print(f"  Max annotations in an image: {max_anns}")
        print(f"  Min annotations in an image: {min_anns}")
    
    # Check for images without annotations
    images_without_anns = len(image_ids) - len(annotations_per_image)
    if images_without_anns > 0:
        print(f"\n⚠️  Warning: {images_without_anns} images have no annotations")
    
    print(f"\n✅ Annotations are valid for SAM3 training!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Verify COCO annotations for SAM3 training"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="workspace/data/mirrors/sam3_format",
        help="Dataset directory containing annotations/"
    )
    parser.add_argument(
        "--splits",
        nargs='+',
        default=['train', 'val'],
        help="Splits to verify (e.g., train val)"
    )
    
    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir)
    annotations_dir = dataset_dir / "annotations"
    
    if not annotations_dir.exists():
        print(f"❌ ERROR: Annotations directory not found: {annotations_dir}")
        return
    
    print("=" * 60)
    print("SAM3 COCO Annotations Verification")
    print("=" * 60)
    
    all_valid = True
    for split in args.splits:
        ann_file = annotations_dir / f"instances_{split}.json"
        if not ann_file.exists():
            print(f"\n⚠️  Skipping {split}: {ann_file} not found")
            continue
        
        if not verify_coco_annotations(ann_file):
            all_valid = False
    
    print("\n" + "=" * 60)
    if all_valid:
        print("✅ ALL ANNOTATIONS VALID - Ready for SAM3 training!")
    else:
        print("❌ SOME ANNOTATIONS INVALID - Please fix errors above")
    print("=" * 60)


if __name__ == "__main__":
    main()
