"""
U-Net Semantic Segmentation Package

A comprehensive, production-ready implementation of U-Net for semantic segmentation
with support for BDD100K and KITTI datasets, video processing, instance extraction,
and advanced deep learning features.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Core imports
from .models import UNet
from .training import SegmentationTrainer
from .inference import SegmentationInference
from .utils import SegmentationVisualizer, SegmentationMetrics

# Dataset imports
from .data import (
    SegmentationDataset,
    list_available_datasets,
    create_sample_dataset_for_testing
)

# Try to import real-world datasets if available
try:
    from .data import BDD100KSegmentationDataset as BDD100KDataset
    from .data import KITTISegmentationDataset as KITTIDataset
except ImportError:
    BDD100KDataset = None
    KITTIDataset = None

# Video processing imports
from .video_processing import VideoFrameIterator, TensorFrameBatcher

# Instance extraction imports
from .instance_extraction import InstanceExtractor

__all__ = [
    # Core classes
    "UNet",
    "SegmentationTrainer", 
    "SegmentationInference",
    "SegmentationVisualizer",
    "SegmentationMetrics",
    
    # Datasets
    "SegmentationDataset",
    "BDD100KDataset", 
    "KITTIDataset",
    "list_available_datasets",
    "create_sample_dataset_for_testing",
    
    # Video processing
    "VideoFrameIterator",
    "TensorFrameBatcher",
    
    # Instance extraction
    "InstanceExtractor",
]
