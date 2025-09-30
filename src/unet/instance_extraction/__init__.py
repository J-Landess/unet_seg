"""
Instance Extraction Package

This package provides algorithms for extracting individual instances from semantic segmentation masks.
Currently supports Watershed and Connected Components algorithms, with extensibility for future SOLO model integration.

Usage:
    from instance_extraction import InstanceExtractor
    from instance_extraction.algorithms import WatershedExtractor, ConnectedComponentsExtractor
    
    # Create extractor
    extractor = InstanceExtractor(algorithm='watershed')
    
    # Extract instances from semantic mask
    instances = extractor.extract_instances(semantic_mask)
    
    # Visualize results
    extractor.visualize_instances(instances, original_image)
"""

from .core import InstanceExtractor
from .algorithms import WatershedExtractor, ConnectedComponentsExtractor
from .visualization import InstanceVisualizer

__version__ = "0.1.0"
__all__ = [
    "InstanceExtractor",
    "WatershedExtractor", 
    "ConnectedComponentsExtractor",
    "InstanceVisualizer"
]
