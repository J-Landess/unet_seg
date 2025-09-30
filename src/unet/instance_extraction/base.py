"""
Base classes for instance extraction algorithms
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple


class BaseInstanceExtractor(ABC):
    """
    Abstract base class for instance extraction algorithms
    """
    
    @abstractmethod
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
            target_classes: List of class IDs to extract instances for
            min_instance_size: Minimum size for valid instances
            
        Returns:
            Dictionary with instance extraction results
        """
        pass
    
    def _filter_instances_by_size(
        self, 
        instance_mask: np.ndarray, 
        min_size: int
    ) -> Tuple[np.ndarray, Dict[int, int]]:
        """
        Filter instances by minimum size
        
        Args:
            instance_mask: Instance mask with unique IDs
            min_size: Minimum instance size
            
        Returns:
            Filtered instance mask and updated class mapping
        """
        unique_ids = np.unique(instance_mask)
        filtered_mask = instance_mask.copy()
        new_class_mapping = {}
        new_instance_id = 1
        
        for instance_id in unique_ids:
            if instance_id == 0:  # Skip background
                continue
                
            mask = instance_mask == instance_id
            size = np.sum(mask)
            
            if size >= min_size:
                # Keep this instance
                filtered_mask[filtered_mask == instance_id] = new_instance_id
                new_class_mapping[new_instance_id] = self._get_class_for_instance(instance_id)
                new_instance_id += 1
            else:
                # Remove this instance
                filtered_mask[mask] = 0
        
        return filtered_mask, new_class_mapping
    
    def _get_class_for_instance(self, instance_id: int) -> int:
        """
        Get class ID for a given instance ID (to be implemented by subclasses)
        """
        # This should be implemented by subclasses based on their specific logic
        return 0  # Default to background class
