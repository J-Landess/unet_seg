"""
Post-processing utilities for instance segmentation
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from scipy import ndimage


class InstancePostProcessor:
    """
    Post-processing utilities for instance segmentation results
    """
    
    def __init__(self):
        """Initialize post-processor"""
        pass
    
    def filter_instances_by_size(
        self,
        instances: Dict[str, np.ndarray],
        min_size: int = 100,
        max_size: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Filter instances by size
        
        Args:
            instances: Instance segmentation results
            min_size: Minimum instance size in pixels
            max_size: Maximum instance size in pixels (None = no limit)
            
        Returns:
            Filtered instance results
        """
        instance_mask = instances['instances']
        class_mapping = instances['class_mapping']
        
        # Get unique instances
        unique_instances = np.unique(instance_mask)
        unique_instances = unique_instances[unique_instances != 0]  # Remove background
        
        filtered_mask = instance_mask.copy()
        new_class_mapping = {}
        new_instance_id = 1
        
        for instance_id in unique_instances:
            mask = instance_mask == instance_id
            size = np.sum(mask)
            
            # Check size constraints
            if size >= min_size and (max_size is None or size <= max_size):
                # Keep this instance
                filtered_mask[filtered_mask == instance_id] = new_instance_id
                new_class_mapping[new_instance_id] = class_mapping.get(instance_id, 0)
                new_instance_id += 1
            else:
                # Remove this instance
                filtered_mask[mask] = 0
        
        return {
            'instances': filtered_mask,
            'class_mapping': new_class_mapping,
            'instance_count': instances.get('instance_count', {})
        }
    
    def filter_instances_by_class(
        self,
        instances: Dict[str, np.ndarray],
        target_classes: List[int]
    ) -> Dict[str, np.ndarray]:
        """
        Filter instances by class
        
        Args:
            instances: Instance segmentation results
            target_classes: List of class IDs to keep
            
        Returns:
            Filtered instance results
        """
        instance_mask = instances['instances']
        class_mapping = instances['class_mapping']
        
        # Create mask for target classes
        target_mask = np.zeros_like(instance_mask, dtype=bool)
        for instance_id, class_id in class_mapping.items():
            if class_id in target_classes:
                target_mask |= (instance_mask == instance_id)
        
        # Filter instances
        filtered_mask = instance_mask.copy()
        filtered_mask[~target_mask] = 0
        
        # Update class mapping
        filtered_class_mapping = {
            instance_id: class_id 
            for instance_id, class_id in class_mapping.items()
            if class_id in target_classes
        }
        
        return {
            'instances': filtered_mask,
            'class_mapping': filtered_class_mapping,
            'instance_count': instances.get('instance_count', {})
        }
    
    def merge_small_instances(
        self,
        instances: Dict[str, np.ndarray],
        max_size: int = 50,
        merge_strategy: str = 'nearest'
    ) -> Dict[str, np.ndarray]:
        """
        Merge small instances with nearby larger instances
        
        Args:
            instances: Instance segmentation results
            max_size: Maximum size for instances to be merged
            merge_strategy: Strategy for merging ('nearest', 'largest')
            
        Returns:
            Processed instance results
        """
        instance_mask = instances['instances']
        class_mapping = instances['class_mapping']
        
        # Get unique instances
        unique_instances = np.unique(instance_mask)
        unique_instances = unique_instances[unique_instances != 0]  # Remove background
        
        processed_mask = instance_mask.copy()
        processed_class_mapping = class_mapping.copy()
        
        # Find small instances
        small_instances = []
        for instance_id in unique_instances:
            mask = instance_mask == instance_id
            size = np.sum(mask)
            if size <= max_size:
                small_instances.append(instance_id)
        
        # Merge small instances
        for small_instance_id in small_instances:
            if small_instance_id not in processed_class_mapping:
                continue
            
            # Find nearest or largest instance of the same class
            target_instance_id = self._find_merge_target(
                small_instance_id, 
                instance_mask, 
                class_mapping, 
                merge_strategy
            )
            
            if target_instance_id is not None:
                # Merge instances
                small_mask = processed_mask == small_instance_id
                processed_mask[small_mask] = target_instance_id
                del processed_class_mapping[small_instance_id]
        
        return {
            'instances': processed_mask,
            'class_mapping': processed_class_mapping,
            'instance_count': instances.get('instance_count', {})
        }
    
    def _find_merge_target(
        self,
        small_instance_id: int,
        instance_mask: np.ndarray,
        class_mapping: Dict[int, int],
        strategy: str
    ) -> Optional[int]:
        """
        Find target instance for merging
        """
        small_class_id = class_mapping.get(small_instance_id, 0)
        small_mask = instance_mask == small_instance_id
        
        # Get instances of the same class
        same_class_instances = [
            instance_id for instance_id, class_id in class_mapping.items()
            if class_id == small_class_id and instance_id != small_instance_id
        ]
        
        if not same_class_instances:
            return None
        
        if strategy == 'nearest':
            # Find nearest instance
            small_centroid = self._get_centroid(small_mask)
            min_distance = float('inf')
            target_instance_id = None
            
            for instance_id in same_class_instances:
                instance_mask_class = instance_mask == instance_id
                if np.sum(instance_mask_class) == 0:
                    continue
                
                centroid = self._get_centroid(instance_mask_class)
                distance = np.linalg.norm(np.array(small_centroid) - np.array(centroid))
                
                if distance < min_distance:
                    min_distance = distance
                    target_instance_id = instance_id
            
            return target_instance_id
        
        elif strategy == 'largest':
            # Find largest instance
            max_size = 0
            target_instance_id = None
            
            for instance_id in same_class_instances:
                instance_mask_class = instance_mask == instance_id
                size = np.sum(instance_mask_class)
                
                if size > max_size:
                    max_size = size
                    target_instance_id = instance_id
            
            return target_instance_id
        
        return None
    
    def _get_centroid(self, mask: np.ndarray) -> Tuple[int, int]:
        """
        Get centroid of a binary mask
        """
        moments = cv2.moments(mask.astype(np.uint8))
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            return (cx, cy)
        else:
            # Fallback to center of bounding box
            coords = np.where(mask)
            if len(coords[0]) > 0:
                return (int(np.mean(coords[1])), int(np.mean(coords[0])))
            return (0, 0)
    
    def smooth_instance_boundaries(
        self,
        instances: Dict[str, np.ndarray],
        kernel_size: int = 3
    ) -> Dict[str, np.ndarray]:
        """
        Smooth instance boundaries using morphological operations
        
        Args:
            instances: Instance segmentation results
            kernel_size: Size of morphological kernel
            
        Returns:
            Processed instance results
        """
        instance_mask = instances['instances']
        class_mapping = instances['class_mapping']
        
        # Create kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Smooth each instance
        smoothed_mask = instance_mask.copy()
        unique_instances = np.unique(instance_mask)
        unique_instances = unique_instances[unique_instances != 0]  # Remove background
        
        for instance_id in unique_instances:
            mask = instance_mask == instance_id
            if np.sum(mask) == 0:
                continue
            
            # Apply morphological closing to smooth boundaries
            smoothed_instance = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            
            # Update mask
            smoothed_mask[smoothed_mask == instance_id] = 0
            smoothed_mask[smoothed_instance > 0] = instance_id
        
        return {
            'instances': smoothed_mask,
            'class_mapping': class_mapping,
            'instance_count': instances.get('instance_count', {})
        }
    
    def remove_duplicate_instances(
        self,
        instances: Dict[str, np.ndarray],
        iou_threshold: float = 0.9
    ) -> Dict[str, np.ndarray]:
        """
        Remove duplicate instances based on IoU
        
        Args:
            instances: Instance segmentation results
            iou_threshold: IoU threshold for considering instances as duplicates
            
        Returns:
            Processed instance results
        """
        instance_mask = instances['instances']
        class_mapping = instances['class_mapping']
        
        # Get unique instances
        unique_instances = np.unique(instance_mask)
        unique_instances = unique_instances[unique_instances != 0]  # Remove background
        
        if len(unique_instances) <= 1:
            return instances
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(unique_instances), len(unique_instances)))
        
        for i, instance_id1 in enumerate(unique_instances):
            for j, instance_id2 in enumerate(unique_instances):
                if i >= j:  # Only compute upper triangle
                    continue
                
                mask1 = instance_mask == instance_id1
                mask2 = instance_mask == instance_id2
                
                iou = self._compute_iou(mask1, mask2)
                iou_matrix[i, j] = iou
                iou_matrix[j, i] = iou  # Symmetric
        
        # Find duplicate pairs
        duplicate_pairs = []
        for i in range(len(unique_instances)):
            for j in range(i + 1, len(unique_instances)):
                if iou_matrix[i, j] >= iou_threshold:
                    duplicate_pairs.append((unique_instances[i], unique_instances[j]))
        
        # Remove duplicates (keep the first instance in each pair)
        instances_to_remove = set()
        for instance_id1, instance_id2 in duplicate_pairs:
            instances_to_remove.add(instance_id2)
        
        # Create filtered mask
        filtered_mask = instance_mask.copy()
        filtered_class_mapping = class_mapping.copy()
        
        for instance_id in instances_to_remove:
            filtered_mask[filtered_mask == instance_id] = 0
            if instance_id in filtered_class_mapping:
                del filtered_class_mapping[instance_id]
        
        return {
            'instances': filtered_mask,
            'class_mapping': filtered_class_mapping,
            'instance_count': instances.get('instance_count', {})
        }
    
    def _compute_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """
        Compute Intersection over Union between two binary masks
        """
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        
        if union == 0:
            return 0.0
        
        return intersection / union
