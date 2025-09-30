"""
Metrics for semantic segmentation evaluation
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple


class SegmentationMetrics:
    """
    Metrics calculator for semantic segmentation
    """
    
    def __init__(self, num_classes: int, ignore_index: Optional[int] = None):
        """
        Initialize metrics calculator
        
        Args:
            num_classes: Number of classes
            ignore_index: Index to ignore in calculations
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.total_correct = 0
        self.total_pixels = 0
        self.class_correct = torch.zeros(self.num_classes)
        self.class_total = torch.zeros(self.num_classes)
        self.intersection = torch.zeros(self.num_classes)
        self.union = torch.zeros(self.num_classes)
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update metrics with new predictions and targets
        
        Args:
            predictions: Predicted class indices [B, H, W]
            targets: Ground truth class indices [B, H, W]
        """
        # Flatten tensors
        pred_flat = predictions.view(-1)
        target_flat = targets.view(-1)
        
        # Remove ignored pixels
        if self.ignore_index is not None:
            valid_mask = target_flat != self.ignore_index
            pred_flat = pred_flat[valid_mask]
            target_flat = target_flat[valid_mask]
        
        # Overall accuracy
        correct = (pred_flat == target_flat).sum().item()
        total = len(pred_flat)
        self.total_correct += correct
        self.total_pixels += total
        
        # Per-class accuracy
        for c in range(self.num_classes):
            class_mask = target_flat == c
            if class_mask.sum() > 0:
                self.class_correct[c] += (pred_flat[class_mask] == c).sum().item()
                self.class_total[c] += class_mask.sum().item()
        
        # IoU calculation
        for c in range(self.num_classes):
            pred_c = (pred_flat == c)
            target_c = (target_flat == c)
            
            intersection = (pred_c & target_c).sum().item()
            union = (pred_c | target_c).sum().item()
            
            self.intersection[c] += intersection
            self.union[c] += union
    
    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics
        
        Returns:
            Dictionary of computed metrics
        """
        metrics = {}
        
        # Overall accuracy
        metrics['accuracy'] = self.total_correct / self.total_pixels if self.total_pixels > 0 else 0.0
        
        # Mean accuracy
        class_accuracies = []
        for c in range(self.num_classes):
            if self.class_total[c] > 0:
                class_acc = self.class_correct[c] / self.class_total[c]
                class_accuracies.append(class_acc.item())
        
        metrics['mean_accuracy'] = np.mean(class_accuracies) if class_accuracies else 0.0
        
        # IoU metrics
        ious = []
        for c in range(self.num_classes):
            if self.union[c] > 0:
                iou = self.intersection[c] / self.union[c]
                ious.append(iou.item())
        
        metrics['mean_iou'] = np.mean(ious) if ious else 0.0
        metrics['class_ious'] = ious
        
        return metrics


def pixel_accuracy(predictions: torch.Tensor, targets: torch.Tensor, ignore_index: Optional[int] = None) -> float:
    """
    Calculate pixel-wise accuracy
    
    Args:
        predictions: Predicted class indices [B, H, W]
        targets: Ground truth class indices [B, H, W]
        ignore_index: Index to ignore in calculations
        
    Returns:
        Pixel accuracy
    """
    if ignore_index is not None:
        valid_mask = targets != ignore_index
        predictions = predictions[valid_mask]
        targets = targets[valid_mask]
    
    correct = (predictions == targets).sum().item()
    total = len(predictions)
    
    return correct / total if total > 0 else 0.0


def mean_iou(predictions: torch.Tensor, targets: torch.Tensor, num_classes: int, ignore_index: Optional[int] = None) -> float:
    """
    Calculate mean Intersection over Union (mIoU)
    
    Args:
        predictions: Predicted class indices [B, H, W]
        targets: Ground truth class indices [B, H, W]
        num_classes: Number of classes
        ignore_index: Index to ignore in calculations
        
    Returns:
        Mean IoU
    """
    if ignore_index is not None:
        valid_mask = targets != ignore_index
        predictions = predictions[valid_mask]
        targets = targets[valid_mask]
    
    ious = []
    for c in range(num_classes):
        pred_c = (predictions == c)
        target_c = (targets == c)
        
        intersection = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()
        
        if union > 0:
            iou = intersection / union
            ious.append(iou)
    
    return np.mean(ious) if ious else 0.0


def dice_coefficient(predictions: torch.Tensor, targets: torch.Tensor, num_classes: int, ignore_index: Optional[int] = None) -> float:
    """
    Calculate Dice coefficient
    
    Args:
        predictions: Predicted class indices [B, H, W]
        targets: Ground truth class indices [B, H, W]
        num_classes: Number of classes
        ignore_index: Index to ignore in calculations
        
    Returns:
        Mean Dice coefficient
    """
    if ignore_index is not None:
        valid_mask = targets != ignore_index
        predictions = predictions[valid_mask]
        targets = targets[valid_mask]
    
    dice_scores = []
    for c in range(num_classes):
        pred_c = (predictions == c)
        target_c = (targets == c)
        
        intersection = (pred_c & target_c).sum().item()
        total = pred_c.sum().item() + target_c.sum().item()
        
        if total > 0:
            dice = 2 * intersection / total
            dice_scores.append(dice)
    
    return np.mean(dice_scores) if dice_scores else 0.0


class LossTracker:
    """
    Track training and validation losses
    """
    
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
    
    def update_train(self, loss: float, metrics: Dict[str, float]):
        """Update training loss and metrics"""
        self.train_losses.append(loss)
        self.train_metrics.append(metrics)
    
    def update_val(self, loss: float, metrics: Dict[str, float]):
        """Update validation loss and metrics"""
        self.val_losses.append(loss)
        self.val_metrics.append(metrics)
    
    def get_best_epoch(self, metric_name: str = 'mean_iou') -> int:
        """Get epoch with best validation metric"""
        if not self.val_metrics:
            return 0
        
        best_metric = max(self.val_metrics, key=lambda x: x.get(metric_name, 0))
        return self.val_metrics.index(best_metric)
    
    def get_latest_metrics(self) -> Dict[str, float]:
        """Get latest validation metrics"""
        if not self.val_metrics:
            return {}
        return self.val_metrics[-1]


if __name__ == "__main__":
    # Test metrics
    num_classes = 21
    batch_size = 2
    height, width = 64, 64
    
    # Create dummy predictions and targets
    predictions = torch.randint(0, num_classes, (batch_size, height, width))
    targets = torch.randint(0, num_classes, (batch_size, height, width))
    
    # Test metrics calculator
    metrics_calc = SegmentationMetrics(num_classes)
    metrics_calc.update(predictions, targets)
    results = metrics_calc.compute()
    
    print("Metrics results:")
    for key, value in results.items():
        if key != 'class_ious':
            print(f"{key}: {value:.4f}")
    
    # Test individual functions
    acc = pixel_accuracy(predictions, targets)
    miou = mean_iou(predictions, targets, num_classes)
    dice = dice_coefficient(predictions, targets, num_classes)
    
    print(f"\nPixel Accuracy: {acc:.4f}")
    print(f"Mean IoU: {miou:.4f}")
    print(f"Dice Coefficient: {dice:.4f}")
