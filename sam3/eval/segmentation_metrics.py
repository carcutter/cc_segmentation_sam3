"""
Custom segmentation metrics for mirror detection training.
Computes IoU, Contour IoU, and visualization.
"""

import logging
import os
import random
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from sam3.train.utils.distributed import all_gather, is_main_process


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Compute Intersection over Union (IoU) between prediction and ground truth masks.
    
    Args:
        pred_mask: Binary prediction mask (H, W)
        gt_mask: Binary ground truth mask (H, W)
    
    Returns:
        IoU score
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return float(intersection / union)


def compute_contour_iou(pred_mask: np.ndarray, gt_mask: np.ndarray, 
                       erosion_kernel_size: int = 5, erosion_iterations: int = 2) -> float:
    """
    Compute Contour IoU - IoU computed on the boundary/contour of masks.
    Uses erosion method matching generate_contour_mask.py reference.
    
    Args:
        pred_mask: Binary prediction mask (H, W)
        gt_mask: Binary ground truth mask (H, W)
        erosion_kernel_size: Size of erosion kernel
        erosion_iterations: Number of erosion iterations
    
    Returns:
        Contour IoU score
    """
    # Extract contours
    pred_contour = extract_contour(pred_mask, erosion_kernel_size, erosion_iterations)
    gt_contour = extract_contour(gt_mask, erosion_kernel_size, erosion_iterations)
    
    # Compute IoU on contours
    return compute_iou(pred_contour, gt_contour)


def extract_contour(mask: np.ndarray, erosion_kernel_size: int = 5, erosion_iterations: int = 2) -> np.ndarray:
    """
    Extract contour from binary mask using erosion method (inner boundary).
    This matches the reference implementation from generate_contour_mask.py.
    
    Args:
        mask: Binary mask (H, W)
        erosion_kernel_size: Size of erosion kernel
        erosion_iterations: Number of erosion iterations
    
    Returns:
        Binary contour mask
    """
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # Check if mask is empty
    if np.max(mask_uint8) == 0:
        return np.zeros_like(mask, dtype=bool)
    
    # Erode the mask to create inner boundary
    erosion_kernel = np.ones((erosion_kernel_size, erosion_kernel_size), np.uint8)
    eroded_mask = cv2.erode(mask_uint8, erosion_kernel, iterations=erosion_iterations)
    
    # The contour is the difference between original and eroded mask
    # This only includes pixels that were inside the original mask
    contour_mask = cv2.subtract(mask_uint8, eroded_mask)
    
    return contour_mask.astype(bool)


def create_overlay_visualization(
    image: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    alpha: float = 0.5,
    iou: float = None,
    contour_iou: float = None
) -> np.ndarray:
    """
    Create visualization with two panels:
    - Left: Image with prediction overlay
    - Right: Comparison of GT and prediction (green=correct, red=missing, orange=false positive)
    
    Args:
        image: RGB image (H, W, 3) in range [0, 255]
        pred_mask: Binary prediction mask (H, W)
        gt_mask: Binary ground truth mask (H, W)
        alpha: Transparency for overlay
        iou: IoU score to display on image
        contour_iou: Contour IoU score to display on image
    
    Returns:
        Visualization image (H, 2*W, 3)
    """
    h, w = image.shape[:2]
    
    # Left panel: Image with prediction overlay (blue)
    left_panel = image.copy()
    pred_overlay = np.zeros_like(image)
    pred_overlay[pred_mask] = [0, 0, 255]  # Blue for predictions
    left_panel = cv2.addWeighted(left_panel, 1 - alpha, pred_overlay, alpha, 0)
    
    # Right panel: GT vs Prediction comparison
    right_panel = image.copy()
    comparison_overlay = np.zeros_like(image)
    
    # True Positives (correct predictions) - Green
    tp_mask = np.logical_and(pred_mask, gt_mask)
    comparison_overlay[tp_mask] = [0, 255, 0]
    
    # False Negatives (missing predictions) - Red
    fn_mask = np.logical_and(~pred_mask, gt_mask)
    comparison_overlay[fn_mask] = [255, 0, 0]
    
    # False Positives (incorrect predictions) - Orange
    fp_mask = np.logical_and(pred_mask, ~gt_mask)
    comparison_overlay[fp_mask] = [255, 165, 0]
    
    right_panel = cv2.addWeighted(right_panel, 1 - alpha, comparison_overlay, alpha, 0)
    
    # Concatenate horizontally
    visualization = np.concatenate([left_panel, right_panel], axis=1)
    
    # Add text with metrics if provided
    if iou is not None and contour_iou is not None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        text = f"IoU: {iou:.3f}  cIoU: {contour_iou:.3f}"
        
        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Add black background for text readability
        cv2.rectangle(visualization, (10, 10), (text_width + 20, text_height + baseline + 20), (0, 0, 0), -1)
        
        # Add white text
        cv2.putText(visualization, text, (15, text_height + 15), font, font_scale, (255, 255, 255), thickness)
    
    return visualization


class SegmentationMetricsMeter:
    """
    Meter for computing segmentation metrics (IoU, Contour IoU) and logging visualizations to TensorBoard.
    """
    
    def __init__(
        self,
        logger = None,
        num_vis_samples: int = 5,
        erosion_kernel_size: int = 5,
        erosion_iterations: int = 2,
        device: str = "cuda",
        metric_prefix: str = "cc_metrics",
        vis_prefix: str = "cc_vis"
    ):
        """
        Args:
            logger: Trainer logger with TensorBoard writer for logging visualizations
            num_vis_samples: Number of random samples to visualize during validation
            erosion_kernel_size: Size of erosion kernel for contour extraction
            erosion_iterations: Number of erosion iterations for contour extraction
            device: Device for computation
            metric_prefix: Prefix for metric names (e.g., 'cc_metrics' -> 'val/cc_metrics/iou')
            vis_prefix: Prefix for visualization names (e.g., 'cc_vis' -> 'val/cc_vis/sample_0')
        """
        self.logger = logger
        self.num_vis_samples = num_vis_samples
        self.erosion_kernel_size = erosion_kernel_size
        self.erosion_iterations = erosion_iterations
        self.device = device
        self.metric_prefix = metric_prefix
        self.vis_prefix = vis_prefix
        self.phase = None  # Will be set by update() based on batch
        
        self.reset()
        
    def reset(self):
        """Reset all accumulated metrics."""
        self.iou_scores = []
        self.contour_iou_scores = []
        self.visualization_data = []
        
    def update(self, find_stages=None, find_metadatas=None, model=None, batch=None, key=None, phase=None, **kwargs):
        """
        Update metrics with batch predictions (trainer-compatible interface).
        
        Args:
            find_stages: Model outputs (SAM3Output object)
            find_metadatas: Metadata for the batch
            model: The model (not used)
            batch: Batch data containing images and targets
            key: Key for the batch (e.g., 'mirror', 'all')
            phase: Training phase ('train' or 'val')
        """
        # Use phase from trainer, or fall back to key-based detection
        if phase is not None:
            self.phase = phase
        elif key is not None and ('val' in key.lower() or 'test' in key.lower()):
            self.phase = 'val'
        else:
            self.phase = 'train'
        
        if find_stages is None or batch is None:
            return
        
        # Extract the last stage outputs (final predictions)
        if hasattr(find_stages, '__iter__'):
            outputs_list = list(find_stages)
            if len(outputs_list) > 0:
                outputs = outputs_list[-1]  # Use last stage
            else:
                return
        else:
            outputs = find_stages
        
        # Get images from batch
        images = None
        if hasattr(batch, 'image_samples'):
            images = batch.image_samples
        elif hasattr(batch, 'images'):
            images = batch.images
        elif hasattr(batch, 'img_batch'):
            images = batch.img_batch
        elif hasattr(batch, 'raw_images'):
            images = batch.raw_images
        
        # Get targets from batch
        targets = {}
        if hasattr(batch, 'find_targets'):
            find_targets = batch.find_targets
            if len(find_targets) > 0:
                targets = find_targets[0]  # First target
        
        # Call the original update method
        self._update_from_outputs(outputs, targets, images)
    
    def _update_from_outputs(self, outputs: Dict, targets: Dict, images: Optional[torch.Tensor] = None):
        """
        Internal method to update metrics with batch predictions.
        
        Args:
            outputs: Model outputs containing 'pred_masks' (B, 1, H, W) or similar
            targets: Ground truth containing 'masks' (B, N, H, W) or 'semantic_masks' (B, H, W)
            images: Original images (B, 3, H, W) for visualization
        """
        # Debug logging disabled to reduce verbosity
        
        # Extract predictions - prioritize per-instance pred_masks over semantic_seg
        pred_masks = None
        pred_logits = None
        
        if hasattr(outputs, 'pred_masks') and outputs.pred_masks is not None:
            pred_masks = outputs.pred_masks
            pred_logits = outputs.pred_logits if hasattr(outputs, 'pred_logits') else None
        elif isinstance(outputs, dict):
            pred_masks = outputs.get('pred_masks')
            pred_logits = outputs.get('pred_logits')
        
        # Fallback to semantic_seg if no pred_masks
        if pred_masks is None:
            if hasattr(outputs, 'semantic_seg') and outputs.semantic_seg is not None:
                pred_masks = outputs.semantic_seg
            elif isinstance(outputs, dict) and 'semantic_seg' in outputs:
                pred_masks = outputs['semantic_seg']
            else:
                logging.warning("No prediction masks found in outputs")
                return
            
        # Check for ground truth masks - use correct attribute names for BatchedFindTarget
        gt_masks = None
        
        # Check semantic_segments first, but only use if non-empty
        if (hasattr(targets, 'semantic_segments') and targets.semantic_segments is not None 
            and targets.semantic_segments.numel() > 0):
            gt_masks = targets.semantic_segments

        # If semantic_segments is empty, try instance segments
        elif hasattr(targets, 'segments') and targets.segments is not None and targets.segments.numel() > 0:
            # segments shape could be:
            # - [B, num_instances, H, W] for batched data
            # - [num_instances, H, W] for single sample
            if targets.segments.ndim == 4:
                # Batched: [B, num_instances, H, W] -> [B, H, W]
                gt_masks = targets.segments.any(dim=1)  # Merge instances per batch item
            elif targets.segments.ndim == 3:
                # Single sample: [num_instances, H, W] -> [1, H, W]
                gt_masks = targets.segments.any(dim=0, keepdim=True)
            elif targets.segments.ndim == 2:
                # Shape: (H, W) - single mask
                gt_masks = targets.segments.unsqueeze(0)
            else:
                gt_masks = targets.segments

        # Fallback to old attribute names for backward compatibility
        elif hasattr(targets, 'semantic_masks') and targets.semantic_masks is not None and targets.semantic_masks.numel() > 0:
            gt_masks = targets.semantic_masks
        elif hasattr(targets, 'masks') and targets.masks is not None and targets.masks.numel() > 0:
            # Merge instance masks to semantic mask
            gt_masks = targets.masks.any(dim=1)
        elif isinstance(targets, dict):
            # Fallback to dictionary access if targets is a dict
            if 'semantic_masks' in targets and targets['semantic_masks'] is not None:
                gt_masks = targets['semantic_masks']
            elif 'masks' in targets:
                gt_masks = targets['masks'].any(dim=1)
        
        # Skip if no ground truth masks found or they're empty
        if gt_masks is None or gt_masks.numel() == 0:
            logging.warning("No valid ground truth masks found!")
            return
        
        # Convert predictions to binary masks
        # Always apply sigmoid to logits first, regardless of dtype
        pred_masks_binary = (pred_masks.sigmoid() > 0.5)
        
        # If we have per-query predictions with logits, filter by detection score
        if pred_logits is not None and pred_masks_binary.ndim == 4:
            # pred_masks_binary shape: [B, num_queries, H, W]
            # pred_logits shape: [B, num_queries, 1] or [B, num_queries]
            scores = pred_logits.sigmoid().squeeze(-1)  # [B, num_queries]
            threshold = 0.4  # Same threshold as COCO evaluator
            
            # Filter valid detections and merge their masks
            batch_size = pred_masks_binary.shape[0]
            merged_masks = []
            num_valid_total = 0
            for b in range(batch_size):
                valid_mask = scores[b] > threshold  # [num_queries]
                num_valid = valid_mask.sum().item()
                num_valid_total += num_valid
                
                if valid_mask.any():
                    # Merge all valid detection masks with logical OR
                    valid_pred_masks = pred_masks_binary[b, valid_mask]  # [num_valid, H, W]
                    merged_mask = valid_pred_masks.any(dim=0, keepdim=True)  # [1, H, W]
                else:
                    # No valid detections - empty mask
                    merged_mask = torch.zeros(1, pred_masks_binary.shape[2], pred_masks_binary.shape[3], 
                                            dtype=torch.bool, device=pred_masks_binary.device)
                merged_masks.append(merged_mask)
            
            if num_valid_total == 0:
                logging.debug(f"No detections above threshold {threshold} in batch (max score: {scores.max().item():.3f})")
            else:
                logging.debug(f"Found {num_valid_total} valid detections above threshold {threshold}")
            
            pred_masks_binary = torch.cat(merged_masks, dim=0)  # [B, H, W]
        elif pred_masks_binary.ndim == 4:
            # No logits available, fall back to first query
            pred_masks_binary = pred_masks_binary[:, 0, :, :]  # [B, H, W]
        elif pred_masks_binary.ndim == 3 and pred_masks_binary.shape[0] > 1:
            # Already [B, H, W] or needs to select first if mismatch
            pass
        else:
            # Ensure we have batch dimension
            if pred_masks_binary.ndim == 2:
                pred_masks_binary = pred_masks_binary.unsqueeze(0)
            
        gt_masks_binary = gt_masks.bool()
        
        # Additional validation
        if gt_masks_binary.numel() == 0:
            return
        
        # Ensure both tensors have batch dimension
        if pred_masks_binary.ndim == 2:
            pred_masks_binary = pred_masks_binary.unsqueeze(0)
        if gt_masks_binary.ndim == 2:
            gt_masks_binary = gt_masks_binary.unsqueeze(0)
        
        # Resize pred masks to match GT resolution if needed
        if pred_masks_binary.shape[-2:] != gt_masks_binary.shape[-2:]:
            import torch.nn.functional as F

            pred_masks_binary = F.interpolate(
                pred_masks_binary.unsqueeze(1).float(),  # [B, 1, H, W]
                size=gt_masks_binary.shape[-2:],
                mode='nearest'
            ).squeeze(1).bool()  # [B, H, W]
        
        # Check for batch size mismatch
        if pred_masks_binary.shape[0] != gt_masks_binary.shape[0]:
            # During training, log this as it indicates an issue
            if self.phase == 'train':
                logging.warning(
                    f"Training batch mismatch: pred={pred_masks_binary.shape}, gt={gt_masks_binary.shape}. "
                    f"This will cause training metrics to be 0. Check data loading."
                )
            # Skip this batch
            return
        
        batch_size = pred_masks_binary.shape[0]
        
        # Compute metrics for each sample in batch
        for i in range(batch_size):
            pred_np = pred_masks_binary[i].cpu().numpy()
            gt_np = gt_masks_binary[i].cpu().numpy()
            
            # Compute IoU
            iou = compute_iou(pred_np, gt_np)
            self.iou_scores.append(iou)
            
            # Compute Contour IoU
            contour_iou = compute_contour_iou(
                pred_np, gt_np, 
                self.erosion_kernel_size, 
                self.erosion_iterations
            )
            self.contour_iou_scores.append(contour_iou)
            
            # Store data for potential visualization

            if images is not None and len(self.visualization_data) < self.num_vis_samples * 2:
                self.visualization_data.append({
                    'image': images[i].cpu(),
                    'pred_mask': pred_np,
                    'gt_mask': gt_np,
                    'iou': iou,
                    'contour_iou': contour_iou
                })

    
    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics.
        
        Returns:
            Dictionary with metric names and values
        """
        if len(self.iou_scores) == 0:
            return {
                'iou': 0.0,
                'contour_iou': 0.0
            }
        
        mean_iou = np.mean(self.iou_scores)
        mean_contour_iou = np.mean(self.contour_iou_scores)
        
        return {
            'iou': float(mean_iou),
            'contour_iou': float(mean_contour_iou)
        }
    
    def compute_synced(self, epoch: int = 0) -> Dict[str, float]:
        """
        Compute metrics synchronized across all processes.
        
        Args:
            epoch: Current epoch number for visualization logging
        
        Returns:
            Dictionary with synchronized metric values
        """
        # Gather metrics from all processes
        all_iou_scores = all_gather(self.iou_scores)
        all_contour_iou_scores = all_gather(self.contour_iou_scores)
        
        # Flatten lists
        all_iou_scores = [item for sublist in all_iou_scores for item in sublist]
        all_contour_iou_scores = [item for sublist in all_contour_iou_scores for item in sublist]
        
        # Get phase prefix early (needed for both cases)
        phase_prefix = self.phase if self.phase else 'train'
        
        if len(all_iou_scores) == 0:
            # Even with no samples, return with phase prefix to avoid double-wrapping
            return {
                f'{phase_prefix}_{self.metric_prefix}/iou': 0.0,
                f'{phase_prefix}_{self.metric_prefix}/contour_iou': 0.0
            }
        
        mean_iou = np.mean(all_iou_scores)
        mean_contour_iou = np.mean(all_contour_iou_scores)
        
        # Create visualizations on main process and log to TensorBoard
        if is_main_process() and self.logger is not None and self.num_vis_samples > 0:
            # Ensure phase is set for visualizations
            if not self.phase:
                self.phase = phase_prefix
            self._log_visualizations_to_tensorboard(epoch)
        
        return {
            f'{phase_prefix}_{self.metric_prefix}/iou': float(mean_iou),
            f'{phase_prefix}_{self.metric_prefix}/contour_iou': float(mean_contour_iou)
        }
    
    def _log_visualizations_to_tensorboard(self, epoch: int = 0):
        """Create visualizations and log them to TensorBoard.
        
        Args:
            epoch: Current epoch number to use as global_step for TensorBoard
        """
        if len(self.visualization_data) == 0:
            logging.warning("No visualization data collected!")
            return
        
        if not self.logger or not self.logger.tb_logger or not self.logger.tb_logger.writer:
            logging.warning("TensorBoard writer not available, skipping visualizations")
            return
        
        logging.info(f"Logging {min(self.num_vis_samples, len(self.visualization_data))} visualizations to TensorBoard from {len(self.visualization_data)} samples")
        
        # Randomly sample num_vis_samples
        samples_to_vis = random.sample(
            self.visualization_data,
            min(self.num_vis_samples, len(self.visualization_data))
        )
        
        writer = self.logger.tb_logger.writer
        global_step = epoch  # Use epoch for x-axis
        
        for idx, sample in enumerate(samples_to_vis):
            # Denormalize image (assuming normalization with mean=0.5, std=0.5)
            image_tensor = sample['image']
            image_np = image_tensor.permute(1, 2, 0).numpy()
            image_np = ((image_np * 0.5) + 0.5) * 255
            image_np = np.clip(image_np, 0, 255).astype(np.uint8)
            
            # Create visualization with metrics overlay
            vis = create_overlay_visualization(
                image_np,
                sample['pred_mask'],
                sample['gt_mask'],
                iou=sample['iou'],
                contour_iou=sample['contour_iou']
            )
            
            # Convert to tensor format (C, H, W) for TensorBoard
            vis_tensor = torch.from_numpy(vis).permute(2, 0, 1).float() / 255.0
            
            # Log to TensorBoard with unique tag per sample (slider shows epochs)
            phase_prefix = self.phase if self.phase else 'train'
            tag = f"{phase_prefix}_{self.vis_prefix}/sample_{idx}"
            writer.add_image(tag, vis_tensor, global_step=global_step)
            
        phase_prefix = self.phase if self.phase else 'train'
        logging.info(f"Logged visualizations to TensorBoard under {phase_prefix}_{self.vis_prefix}")
