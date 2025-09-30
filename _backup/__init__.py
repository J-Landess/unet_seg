"""
Semantic Segmentation U-Net Package

A modular implementation of U-Net for semantic segmentation using PyTorch.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .models import UNet, create_unet_model
from .data import SegmentationDataset, PascalVOCDataset, create_dataloader
from .training import SegmentationTrainer
from .inference import SegmentationInference
from .utils import (
    SegmentationMetrics,
    pixel_accuracy,
    mean_iou,
    dice_coefficient,
    LossTracker,
    SegmentationVisualizer,
    save_prediction_visualization
)

__all__ = [
    # Models
    'UNet',
    'create_unet_model',
    
    # Data
    'SegmentationDataset',
    'PascalVOCDataset', 
    'create_dataloader',
    
    # Training
    'SegmentationTrainer',
    
    # Inference
    'SegmentationInference',
    
    # Utils
    'SegmentationMetrics',
    'pixel_accuracy',
    'mean_iou', 
    'dice_coefficient',
    'LossTracker',
    'SegmentationVisualizer',
    'save_prediction_visualization'
]
