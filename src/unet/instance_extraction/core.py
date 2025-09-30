"""
Core instance extraction interface and base classes
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import cv2

from .algorithms import WatershedExtractor, ConnectedComponentsExtractor
from .visualization import InstanceVisualizer


class InstanceExtractor:
    """
    Main interface for instance extraction from semantic segmentation masks.
    
    This class provides a unified interface for different instance extraction algorithms
    and is designed to be extensible for future SOLO model integration.
    """
    
    def __init__(self, algorithm: str = 'watershed', **kwargs):
        """
        Initialize instance extractor
        
        Args:
            algorithm: Algorithm to use ('watershed', 'connected_components', or 'solo' for future)
            **kwargs: Additional parameters for the specific algorithm
        """
        self.algorithm = algorithm.lower()
        self.kwargs = kwargs
        
        # Initialize the specific algorithm
        if self.algorithm == 'watershed':
            self.extractor = WatershedExtractor(**kwargs)
        elif self.algorithm == 'connected_components':
            self.extractor = ConnectedComponentsExtractor(**kwargs)
        elif self.algorithm == 'solo':
            # Future SOLO model integration
            raise NotImplementedError("SOLO model integration not yet implemented")
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}. Supported: 'watershed', 'connected_components'")
        
        # Initialize visualizer
        self.visualizer = InstanceVisualizer()
    
    def extract_instances(
        self, 
        semantic_mask: np.ndarray,
        target_classes: Optional[List[int]] = None,
        min_instance_size: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Extract instances from semantic segmentation mask
        
        Args:
            semantic_mask: Semantic segmentation mask with class IDs
            target_classes: List of class IDs to extract instances for (None = all classes)
            min_instance_size: Minimum size for valid instances (in pixels)
            
        Returns:
            Dictionary containing:
            - 'instances': Instance mask with unique IDs for each instance
            - 'class_mapping': Mapping from instance ID to class ID
            - 'instance_count': Number of instances found per class
        """
        return self.extractor.extract_instances(
            semantic_mask=semantic_mask,
            target_classes=target_classes,
            min_instance_size=min_instance_size
        )
    
    def visualize_instances(
        self,
        instances: Dict[str, np.ndarray],
        original_image: Optional[np.ndarray] = None,
        output_path: Optional[Union[str, Path]] = None,
        show_overlay: bool = True,
        show_contours: bool = True,
        show_labels: bool = True
    ) -> np.ndarray:
        """
        Visualize extracted instances
        
        Args:
            instances: Instance extraction results
            original_image: Original RGB image for overlay
            output_path: Path to save visualization
            show_overlay: Whether to show overlay on original image
            
        Returns:
            Visualization image
        """
        return self.visualizer.visualize_instances(
            instances=instances,
            original_image=original_image,
            output_path=output_path,
            show_overlay=show_overlay,
            show_contours=show_contours,
            show_labels=show_labels
        )
    
    def get_instance_statistics(self, instances: Dict[str, np.ndarray]) -> Dict[str, any]:
        """
        Get statistics about extracted instances
        
        Args:
            instances: Instance extraction results
            
        Returns:
            Dictionary with instance statistics
        """
        instance_mask = instances['instances']
        class_mapping = instances['class_mapping']
        
        stats = {
            'total_instances': len(np.unique(instance_mask)) - 1,  # Exclude background (0)
            'instances_per_class': {},
            'class_distribution': {},
            'instance_sizes': []
        }
        
        # Count instances per class
        for instance_id, class_id in class_mapping.items():
            if instance_id == 0:  # Skip background
                continue
                
            mask = instance_mask == instance_id
            size = np.sum(mask)
            stats['instance_sizes'].append(size)
            
            if class_id not in stats['instances_per_class']:
                stats['instances_per_class'][class_id] = 0
            stats['instances_per_class'][class_id] += 1
        
        # Class distribution
        for class_id, count in stats['instances_per_class'].items():
            stats['class_distribution'][class_id] = count / stats['total_instances']
        
        # Size statistics
        if stats['instance_sizes']:
            stats['avg_instance_size'] = np.mean(stats['instance_sizes'])
            stats['min_instance_size'] = np.min(stats['instance_sizes'])
            stats['max_instance_size'] = np.max(stats['instance_sizes'])
        
        return stats


