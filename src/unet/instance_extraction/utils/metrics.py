"""
Metrics for instance segmentation evaluation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


class InstanceMetrics:
    """
    Metrics calculator for instance segmentation evaluation
    """
    
    def __init__(self, iou_threshold: float = 0.5):
        """
        Initialize metrics calculator
        
        Args:
            iou_threshold: IoU threshold for matching instances
        """
        self.iou_threshold = iou_threshold
    
    def compute_metrics(
        self,
        pred_instances: Dict[str, np.ndarray],
        gt_instances: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Compute instance segmentation metrics
        
        Args:
            pred_instances: Predicted instances
            gt_instances: Ground truth instances
            
        Returns:
            Dictionary of computed metrics
        """
        pred_mask = pred_instances['instances']
        pred_class_mapping = pred_instances['class_mapping']
        
        gt_mask = gt_instances['instances']
        gt_class_mapping = gt_instances['class_mapping']
        
        # Get unique instances
        pred_instance_ids = [id for id in pred_class_mapping.keys() if id != 0]
        gt_instance_ids = [id for id in gt_class_mapping.keys() if id != 0]
        
        if len(pred_instance_ids) == 0 and len(gt_instance_ids) == 0:
            return {
                'precision': 1.0,
                'recall': 1.0,
                'f1': 1.0,
                'mAP': 1.0,
                'num_pred_instances': 0,
                'num_gt_instances': 0,
                'num_matched': 0
            }
        
        if len(pred_instance_ids) == 0:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'mAP': 0.0,
                'num_pred_instances': 0,
                'num_gt_instances': len(gt_instance_ids),
                'num_matched': 0
            }
        
        if len(gt_instance_ids) == 0:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'mAP': 0.0,
                'num_pred_instances': len(pred_instance_ids),
                'num_gt_instances': 0,
                'num_matched': 0
            }
        
        # Compute IoU matrix
        iou_matrix = self._compute_iou_matrix(pred_mask, gt_mask, pred_instance_ids, gt_instance_ids)
        
        # Find optimal matching using Hungarian algorithm
        matched_pairs, unmatched_pred, unmatched_gt = self._match_instances(
            iou_matrix, pred_instance_ids, gt_instance_ids
        )
        
        # Compute metrics
        num_matched = len(matched_pairs)
        num_pred = len(pred_instance_ids)
        num_gt = len(gt_instance_ids)
        
        precision = num_matched / num_pred if num_pred > 0 else 0.0
        recall = num_matched / num_gt if num_gt > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Compute mAP
        map_score = self._compute_map(matched_pairs, iou_matrix, pred_instance_ids, gt_instance_ids)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mAP': map_score,
            'num_pred_instances': num_pred,
            'num_gt_instances': num_gt,
            'num_matched': num_matched
        }
    
    def _compute_iou_matrix(
        self,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
        pred_instance_ids: List[int],
        gt_instance_ids: List[int]
    ) -> np.ndarray:
        """
        Compute IoU matrix between predicted and ground truth instances
        """
        iou_matrix = np.zeros((len(pred_instance_ids), len(gt_instance_ids)))
        
        for i, pred_id in enumerate(pred_instance_ids):
            for j, gt_id in enumerate(gt_instance_ids):
                pred_instance = (pred_mask == pred_id).astype(np.uint8)
                gt_instance = (gt_mask == gt_id).astype(np.uint8)
                
                iou = self._compute_iou(pred_instance, gt_instance)
                iou_matrix[i, j] = iou
        
        return iou_matrix
    
    def _compute_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """
        Compute Intersection over Union between two binary masks
        """
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _match_instances(
        self,
        iou_matrix: np.ndarray,
        pred_instance_ids: List[int],
        gt_instance_ids: List[int]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match instances using Hungarian algorithm
        """
        # Only consider pairs with IoU above threshold
        valid_pairs = iou_matrix >= self.iou_threshold
        
        if not np.any(valid_pairs):
            return [], pred_instance_ids, gt_instance_ids
        
        # Create cost matrix (negative IoU for minimization)
        cost_matrix = -iou_matrix.copy()
        cost_matrix[~valid_pairs] = 1e6  # High cost for invalid pairs
        
        # Solve assignment problem
        pred_indices, gt_indices = linear_sum_assignment(cost_matrix)
        
        # Filter out invalid matches
        matched_pairs = []
        for pred_idx, gt_idx in zip(pred_indices, gt_indices):
            if valid_pairs[pred_idx, gt_idx]:
                matched_pairs.append((pred_instance_ids[pred_idx], gt_instance_ids[gt_idx]))
        
        # Find unmatched instances
        matched_pred_ids = [pair[0] for pair in matched_pairs]
        matched_gt_ids = [pair[1] for pair in matched_pairs]
        
        unmatched_pred = [id for id in pred_instance_ids if id not in matched_pred_ids]
        unmatched_gt = [id for id in gt_instance_ids if id not in matched_gt_ids]
        
        return matched_pairs, unmatched_pred, unmatched_gt
    
    def _compute_map(
        self,
        matched_pairs: List[Tuple[int, int]],
        iou_matrix: np.ndarray,
        pred_instance_ids: List[int],
        gt_instance_ids: List[int]
    ) -> float:
        """
        Compute mean Average Precision (mAP)
        """
        if len(matched_pairs) == 0:
            return 0.0
        
        # Get IoU scores for matched pairs
        iou_scores = []
        for pred_id, gt_id in matched_pairs:
            pred_idx = pred_instance_ids.index(pred_id)
            gt_idx = gt_instance_ids.index(gt_id)
            iou_scores.append(iou_matrix[pred_idx, gt_idx])
        
        # Compute AP as average IoU of matched instances
        return np.mean(iou_scores)
    
    def compute_instance_statistics(self, instances: Dict[str, np.ndarray]) -> Dict[str, any]:
        """
        Compute statistics for instance segmentation results
        
        Args:
            instances: Instance segmentation results
            
        Returns:
            Dictionary with instance statistics
        """
        instance_mask = instances['instances']
        class_mapping = instances['class_mapping']
        
        # Get unique instances
        unique_instances = np.unique(instance_mask)
        unique_instances = unique_instances[unique_instances != 0]  # Remove background
        
        if len(unique_instances) == 0:
            return {
                'num_instances': 0,
                'instances_per_class': {},
                'instance_sizes': [],
                'avg_instance_size': 0,
                'min_instance_size': 0,
                'max_instance_size': 0
            }
        
        # Compute statistics
        instance_sizes = []
        instances_per_class = {}
        
        for instance_id in unique_instances:
            mask = instance_mask == instance_id
            size = np.sum(mask)
            instance_sizes.append(size)
            
            class_id = class_mapping.get(instance_id, 0)
            if class_id not in instances_per_class:
                instances_per_class[class_id] = 0
            instances_per_class[class_id] += 1
        
        return {
            'num_instances': len(unique_instances),
            'instances_per_class': instances_per_class,
            'instance_sizes': instance_sizes,
            'avg_instance_size': np.mean(instance_sizes),
            'min_instance_size': np.min(instance_sizes),
            'max_instance_size': np.max(instance_sizes)
        }
