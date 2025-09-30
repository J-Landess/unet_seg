"""
Watershed algorithm for instance segmentation
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from scipy import ndimage
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.morphology import disk, opening, closing

from ..base import BaseInstanceExtractor


class WatershedExtractor(BaseInstanceExtractor):
    """
    Watershed-based instance extraction from semantic segmentation masks.
    
    This algorithm uses the watershed transform to separate touching objects
    of the same class by finding local maxima and using them as seeds.
    """
    
    def __init__(
        self,
        min_distance: int = 10,
        threshold_abs: float = 0.3,
        min_separation: int = 5,
        compactness: float = 0.0,
        watershed_line: bool = False
    ):
        """
        Initialize Watershed extractor
        
        Args:
            min_distance: Minimum distance between peaks for peak detection
            threshold_abs: Minimum absolute intensity for peaks
            min_separation: Minimum separation between watershed seeds
            compactness: Compactness parameter for watershed (0 = no compactness)
            watershed_line: Whether to include watershed lines in result
        """
        self.min_distance = min_distance
        self.threshold_abs = threshold_abs
        self.min_separation = min_separation
        self.compactness = compactness
        self.watershed_line = watershed_line
    
    def extract_instances(
        self,
        semantic_mask: np.ndarray,
        target_classes: Optional[List[int]] = None,
        min_instance_size: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Extract instances using watershed algorithm
        
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
        kernel = disk(2)
        cleaned = opening(mask, kernel)
        
        # Fill small holes
        cleaned = closing(cleaned, kernel)
        
        return cleaned
    
    def _extract_class_instances(
        self, 
        class_mask: np.ndarray, 
        class_id: int,
        start_instance_id: int
    ) -> Dict[int, np.ndarray]:
        """
        Extract instances for a specific class using watershed
        
        Args:
            class_mask: Binary mask for the class
            class_id: Class ID
            start_instance_id: Starting instance ID
            
        Returns:
            Dictionary mapping instance ID to instance mask
        """
        # Compute distance transform
        distance = ndimage.distance_transform_edt(class_mask)
        
        # Find local maxima as seeds
        local_maxima = peak_local_max(
            distance,
            min_distance=self.min_distance,
            threshold_abs=self.threshold_abs * np.max(distance)
        )
        
        if len(local_maxima) == 0 or len(local_maxima[0]) == 0:
            # No local maxima found, treat as single instance
            if np.sum(class_mask) > 0:
                return {start_instance_id: class_mask}
            return {}
        
        # Create markers for watershed
        markers = np.zeros_like(distance, dtype=np.int32)
        if len(local_maxima) > 0:
            for i, (y, x) in enumerate(local_maxima):
                markers[y, x] = i + 1
        
        # Apply watershed
        labels = watershed(
            -distance,  # Negative distance for watershed
            markers,
            mask=class_mask,
            compactness=self.compactness,
            watershed_line=self.watershed_line
        )
        
        # Extract individual instances
        instances = {}
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            if label == 0:  # Skip background
                continue
                
            instance_mask = (labels == label).astype(np.uint8)
            
            # Check if instance is large enough
            if np.sum(instance_mask) >= self.min_separation:
                instances[start_instance_id] = instance_mask
                start_instance_id += 1
        
        return instances
    
    def _get_class_for_instance(self, instance_id: int) -> int:
        """
        Get class ID for a given instance ID
        This is handled in the main extraction method
        """
        # This is handled in the main extraction method
        return 0
