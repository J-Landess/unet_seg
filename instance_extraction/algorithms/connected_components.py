"""
Connected Components algorithm for instance segmentation
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from scipy import ndimage

from ..base import BaseInstanceExtractor


class ConnectedComponentsExtractor(BaseInstanceExtractor):
    """
    Connected Components-based instance extraction from semantic segmentation masks.
    
    This algorithm uses connected component analysis to identify separate instances
    of the same class. It's simpler and faster than watershed but may not separate
    touching objects as effectively.
    """
    
    def __init__(
        self,
        connectivity: int = 8,
        min_area: int = 50,
        max_area: Optional[int] = None
    ):
        """
        Initialize Connected Components extractor
        
        Args:
            connectivity: Connectivity for connected components (4 or 8)
            min_area: Minimum area for valid instances
            max_area: Maximum area for valid instances (None = no limit)
        """
        self.connectivity = connectivity
        self.min_area = min_area
        self.max_area = max_area
    
    def extract_instances(
        self,
        semantic_mask: np.ndarray,
        target_classes: Optional[List[int]] = None,
        min_instance_size: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Extract instances using connected components algorithm
        
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
        if target_classes is None:
            target_classes = list(np.unique(semantic_mask))
            if 0 in target_classes:
                target_classes.remove(0)  # Remove background
        
        # Initialize result mask
        instance_mask = np.zeros_like(semantic_mask, dtype=np.int32)
        class_mapping = {}
        instance_count = {}
        current_instance_id = 1
        
        # Process each target class
        for class_id in target_classes:
            if class_id == 0:  # Skip background
                continue
                
            # Get binary mask for this class
            class_mask = (semantic_mask == class_id).astype(np.uint8)
            
            if np.sum(class_mask) == 0:
                continue
            
            # Apply morphological operations to clean up the mask
            cleaned_mask = self._clean_mask(class_mask)
            
            # Extract instances for this class
            class_instances = self._extract_class_instances(
                cleaned_mask, 
                class_id, 
                current_instance_id
            )
            
            # Update results
            for instance_id, instance_mask_class in class_instances.items():
                instance_mask[instance_mask_class > 0] = instance_id
                class_mapping[instance_id] = class_id
                current_instance_id += 1
            
            instance_count[class_id] = len(class_instances)
        
        # Filter instances by size
        filtered_mask, filtered_mapping = self._filter_instances_by_size(
            instance_mask, min_instance_size
        )
        
        return {
            'instances': filtered_mask,
            'class_mapping': filtered_mapping,
            'instance_count': instance_count
        }
    
    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Clean up binary mask using morphological operations
        
        Args:
            mask: Binary mask
            
        Returns:
            Cleaned binary mask
        """
        # Remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Fill small holes
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def _extract_class_instances(
        self, 
        class_mask: np.ndarray, 
        class_id: int,
        start_instance_id: int
    ) -> Dict[int, np.ndarray]:
        """
        Extract instances for a specific class using connected components
        
        Args:
            class_mask: Binary mask for the class
            class_id: Class ID
            start_instance_id: Starting instance ID
            
        Returns:
            Dictionary mapping instance ID to instance mask
        """
        # Find connected components
        num_labels, labels = cv2.connectedComponents(
            class_mask, 
            connectivity=self.connectivity
        )
        
        instances = {}
        current_instance_id = start_instance_id
        
        # Process each connected component
        for label in range(1, num_labels):  # Skip background (label 0)
            component_mask = (labels == label).astype(np.uint8)
            area = np.sum(component_mask)
            
            # Check area constraints
            if area < self.min_area:
                continue
            if self.max_area is not None and area > self.max_area:
                continue
            
            instances[current_instance_id] = component_mask
            current_instance_id += 1
        
        return instances
    
    def _get_class_for_instance(self, instance_id: int) -> int:
        """
        Get class ID for a given instance ID
        This is handled in the main extraction method
        """
        # This is handled in the main extraction method
        return 0
