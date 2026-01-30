#!/usr/bin/env python3
"""
CLIP-based Vehicle Type Classifier

This script uses CLIP (from Hugging Face) to classify images into vehicle categories:
- car
- truck
- motorcycle
- boat
- other (non-vehicle or unrecognized)

It processes multiple batches, saves classification results to JSON, and creates
a validation split (5% per category per batch).
"""

import os
import json
import random
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


# ========================= CONFIGURATION =========================

# Batch configuration
ROOT_FOLDER = "/home/raul/workspace/data"
BATCH_NAME_LIST = [
    "car_segmentation/kw2551_car_segmentation",
    "car_segmentation/kw2552_car_interior_segmentation",
    "car_segmentation/kw2553_car_segmentation",
    "car_segmentation/kw2602_car_segmentation"
]

# Task name for output organization
TASK_NAME = "vehicle_classification"

# Output directory
OUTPUT_DIR = Path(f"/home/raul/workspace/data/car_segmentation/classification_results")

# CLIP model configuration
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# Classification categories
VEHICLE_CATEGORIES = ["car", "truck", "motorcycle", "boat"]

# Classification prompts (CLIP works better with descriptive prompts)
CLASSIFICATION_PROMPTS = [
    "a photo of a car",
    "a photo of a truck",
    "a photo of a motorcycle",
    "a photo of a boat",
    "a photo of something else"
]

# Confidence threshold - below this, classify as "other"
CONFIDENCE_THRESHOLD = 0.3

# Validation split percentage
VALIDATION_SPLIT_PERCENT = 5  # 5%

# Random seed for reproducibility
RANDOM_SEED = 42

# =================================================================


def load_clip_model(model_name: str, device: str):
    """Load CLIP model and processor from Hugging Face."""
    print(f"Loading CLIP model: {model_name}")
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    print(f"Model loaded on device: {device}")
    return model, processor


def classify_image(image_path: Path, model, processor, device: str) -> dict:
    """
    Classify a single image using CLIP.
    
    Returns:
        dict with 'category', 'confidence', and 'all_scores'
    """
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        
        # Process inputs
        inputs = processor(
            text=CLASSIFICATION_PROMPTS,
            images=image,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
        
        # Get scores for each category
        probs_np = probs.cpu().numpy()[0]
        
        # Map to categories (last one is "other")
        categories = VEHICLE_CATEGORIES + ["other"]
        scores = {cat: float(probs_np[i]) for i, cat in enumerate(categories)}
        
        # Get best category (excluding "other" unless it's the highest)
        best_idx = probs_np.argmax()
        best_category = categories[best_idx]
        best_confidence = float(probs_np[best_idx])
        
        # If confidence is below threshold for vehicle categories, mark as "other"
        if best_category != "other" and best_confidence < CONFIDENCE_THRESHOLD:
            best_category = "other"
        
        return {
            "category": best_category,
            "confidence": best_confidence,
            "all_scores": scores
        }
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return {
            "category": "error",
            "confidence": 0.0,
            "all_scores": {},
            "error": str(e)
        }


def process_batch(batch_name: str, model, processor, device: str) -> dict:
    """
    Process all images in a batch and classify them.
    
    Returns:
        dict with classification results for the batch
    """
    folder_path = f"{ROOT_FOLDER}/{batch_name}"
    images_dir = Path(f"{folder_path}/raw")
    
    print(f"\n{'='*60}")
    print(f"Processing batch: {batch_name}")
    print(f"Images directory: {images_dir}")
    
    if not images_dir.exists():
        print(f"WARNING: Images directory not found, skipping batch")
        return None
    
    # Get all image files
    image_files = sorted(
        list(images_dir.glob("*.jpg")) + 
        list(images_dir.glob("*.png")) +
        list(images_dir.glob("*.jpeg"))
    )
    
    print(f"Found {len(image_files)} images")
    
    if len(image_files) == 0:
        return None
    
    # Initialize results structure
    batch_results = {
        "batch_name": batch_name,
        "total_images": len(image_files),
        "categories": defaultdict(list),
        "category_counts": defaultdict(int),
        "detailed_results": {}
    }
    
    # Process each image
    for img_path in tqdm(image_files, desc=f"Classifying {batch_name.split('/')[-1]}"):
        result = classify_image(img_path, model, processor, device)
        
        category = result["category"]
        filename = img_path.name
        
        # Store in categories list
        batch_results["categories"][category].append(filename)
        batch_results["category_counts"][category] += 1
        
        # Store detailed result
        batch_results["detailed_results"][filename] = {
            "category": category,
            "confidence": result["confidence"],
            "scores": result.get("all_scores", {})
        }
    
    # Convert defaultdicts to regular dicts for JSON serialization
    batch_results["categories"] = dict(batch_results["categories"])
    batch_results["category_counts"] = dict(batch_results["category_counts"])
    
    # Print summary
    print(f"\nBatch Summary:")
    for cat, count in sorted(batch_results["category_counts"].items()):
        percentage = (count / batch_results["total_images"]) * 100
        print(f"  {cat}: {count} ({percentage:.1f}%)")
    
    return batch_results


def create_validation_split(classification_results: dict, split_percent: float = 5.0) -> dict:
    """
    Create validation split by selecting split_percent% of images per category per batch.
    
    Args:
        classification_results: Full classification results
        split_percent: Percentage of images to select for validation
    
    Returns:
        dict with validation file lists per batch
    """
    print(f"\n{'='*60}")
    print(f"Creating Validation Split ({split_percent}% per category per batch)")
    print(f"{'='*60}")
    
    random.seed(RANDOM_SEED)
    
    validation_split = {
        "split_percent": split_percent,
        "random_seed": RANDOM_SEED,
        "batches": {}
    }
    
    total_val_images = 0
    
    for batch_name, batch_data in classification_results["batches"].items():
        if batch_data is None:
            continue
        
        batch_validation = {
            "files_by_category": {},
            "all_validation_files": [],
            "category_counts": {}
        }
        
        categories = batch_data.get("categories", {})
        
        print(f"\nBatch: {batch_name}")
        
        for category, files in categories.items():
            if not files:
                continue
            
            # Calculate number of files to select
            n_files = len(files)
            n_select = max(1, int(n_files * split_percent / 100))
            
            # Randomly select files
            selected_files = random.sample(files, min(n_select, n_files))
            
            batch_validation["files_by_category"][category] = selected_files
            batch_validation["all_validation_files"].extend(selected_files)
            batch_validation["category_counts"][category] = len(selected_files)
            
            print(f"  {category}: {len(selected_files)}/{n_files} selected")
            total_val_images += len(selected_files)
        
        validation_split["batches"][batch_name] = batch_validation
    
    validation_split["total_validation_images"] = total_val_images
    
    print(f"\nTotal validation images: {total_val_images}")
    
    return validation_split


def main():
    """Main function to run CLIP vehicle classification."""
    
    print("="*60)
    print("CLIP Vehicle Type Classifier")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Model: {CLIP_MODEL_NAME}")
    print(f"  Categories: {VEHICLE_CATEGORIES}")
    print(f"  Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"  Validation split: {VALIDATION_SPLIT_PERCENT}%")
    print(f"  Batches: {len(BATCH_NAME_LIST)}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    # Load CLIP model
    model, processor = load_clip_model(CLIP_MODEL_NAME, device)
    
    # Process all batches
    all_results = {
        "task": TASK_NAME,
        "model": CLIP_MODEL_NAME,
        "categories": VEHICLE_CATEGORIES,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "batches": {}
    }
    
    for batch_name in BATCH_NAME_LIST:
        batch_results = process_batch(batch_name, model, processor, device)
        all_results["batches"][batch_name] = batch_results
    
    # Save classification results
    classification_output_path = OUTPUT_DIR / "classification_results.json"
    with open(classification_output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'='*60}")
    print(f"Classification results saved to: {classification_output_path}")
    
    # Print overall summary
    print(f"\n{'='*60}")
    print("Overall Classification Summary")
    print(f"{'='*60}")
    
    total_counts = defaultdict(int)
    total_images = 0
    
    for batch_name, batch_data in all_results["batches"].items():
        if batch_data is None:
            continue
        total_images += batch_data["total_images"]
        for cat, count in batch_data["category_counts"].items():
            total_counts[cat] += count
    
    print(f"\nTotal images processed: {total_images}")
    print("\nCategory distribution across all batches:")
    for cat, count in sorted(total_counts.items()):
        percentage = (count / total_images) * 100 if total_images > 0 else 0
        print(f"  {cat}: {count} ({percentage:.1f}%)")
    
    # Create validation split
    validation_split = create_validation_split(all_results, VALIDATION_SPLIT_PERCENT)
    
    # Save validation split
    validation_output_path = OUTPUT_DIR / "validation_split.json"
    with open(validation_output_path, 'w') as f:
        json.dump(validation_split, f, indent=2)
    print(f"\nValidation split saved to: {validation_output_path}")
    
    # Also save a simple list of validation files per batch
    validation_lists_dir = OUTPUT_DIR / "validation_lists"
    validation_lists_dir.mkdir(exist_ok=True)
    
    for batch_name, batch_val in validation_split["batches"].items():
        # Create a clean filename from batch name
        batch_filename = batch_name.replace("/", "_") + "_validation.txt"
        batch_list_path = validation_lists_dir / batch_filename
        
        with open(batch_list_path, 'w') as f:
            for filename in sorted(batch_val["all_validation_files"]):
                f.write(f"{filename}\n")
        
        print(f"  Saved: {batch_list_path}")
    
    print(f"\n{'='*60}")
    print("Processing complete!")
    print(f"{'='*60}")
    print(f"\nOutput files:")
    print(f"  1. {classification_output_path}")
    print(f"  2. {validation_output_path}")
    print(f"  3. {validation_lists_dir}/*.txt")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()



# info
# Step 1: Classification

# Loads CLIP model (openai/clip-vit-base-patch32) from Hugging Face
# Processes each batch's raw images
# Classifies into categories: car, truck, motorcycle, boat, other
# Saves results to classification_results.json with:
# Files per batch grouped by category
# Confidence scores for each classification
# Category counts and percentages
# Step 2: Validation Split

# Selects 5% of images per category per batch
# Uses random seed (42) for reproducibility
# Saves to validation_split.json
# Also creates simple .txt file lists per batch in validation_lists/
# /home/raul/workspace/data/car_segmentation/classification_results/
# ├── classification_results.json    # Full classification data
# ├── validation_split.json          # Validation split info
# └── validation_lists/
#     ├── car_segmentation_kw2551_car_segmentation_validation.txt
#     ├── car_segmentation_kw2552_car_interior_segmentation_validation.txt
#     └── ...