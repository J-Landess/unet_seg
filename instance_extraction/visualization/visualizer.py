"""
Instance visualization utilities
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


class InstanceVisualizer:
    """
    Visualization utilities for instance segmentation results
    """
    
    def __init__(self, colormap: str = 'tab20'):
        """
        Initialize visualizer
        
        Args:
            colormap: Matplotlib colormap name for instance colors
        """
        self.colormap = colormap
        self._instance_colors = None
        self._class_colors = self._get_class_colors()
    
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
            show_contours: Whether to show instance contours
            show_labels: Whether to show instance labels
            
        Returns:
            Visualization image
        """
        instance_mask = instances['instances']
        class_mapping = instances['class_mapping']
        
        # Create base visualization
        if show_overlay and original_image is not None:
            vis_image = self._create_overlay_visualization(
                instance_mask, class_mapping, original_image
            )
        else:
            vis_image = self._create_mask_visualization(
                instance_mask, class_mapping
            )
        
        # Add contours if requested
        if show_contours:
            vis_image = self._add_contours(vis_image, instance_mask)
        
        # Add labels if requested
        if show_labels:
            vis_image = self._add_labels(vis_image, instance_mask, class_mapping)
        
        # Save if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        
        return vis_image
    
    def _create_overlay_visualization(
        self, 
        instance_mask: np.ndarray, 
        class_mapping: Dict[int, int],
        original_image: np.ndarray
    ) -> np.ndarray:
        """
        Create overlay visualization on original image
        """
        # Ensure original image is RGB
        if len(original_image.shape) == 3 and original_image.shape[2] == 3:
            if original_image.dtype == np.uint8:
                overlay = original_image.copy()
            else:
                overlay = (original_image * 255).astype(np.uint8)
        else:
            overlay = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        
        # Resize instance mask to match original image
        if instance_mask.shape != overlay.shape[:2]:
            instance_mask = cv2.resize(
                instance_mask.astype(np.uint8), 
                (overlay.shape[1], overlay.shape[0]), 
                interpolation=cv2.INTER_NEAREST
            )
        
        # Create colored instance mask
        colored_mask = self._colorize_instances(instance_mask, class_mapping)
        
        # Blend with original image
        alpha = 0.6
        overlay = cv2.addWeighted(overlay, 1 - alpha, colored_mask, alpha, 0)
        
        return overlay
    
    def _create_mask_visualization(
        self, 
        instance_mask: np.ndarray, 
        class_mapping: Dict[int, int]
    ) -> np.ndarray:
        """
        Create mask-only visualization
        """
        return self._colorize_instances(instance_mask, class_mapping)
    
    def _colorize_instances(
        self, 
        instance_mask: np.ndarray, 
        class_mapping: Dict[int, int]
    ) -> np.ndarray:
        """
        Colorize instance mask with distinct colors
        """
        h, w = instance_mask.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Get unique instance IDs
        unique_instances = np.unique(instance_mask)
        
        for instance_id in unique_instances:
            if instance_id == 0:  # Skip background
                continue
            
            # Get color for this instance
            color = self._get_instance_color(instance_id)
            
            # Apply color to instance pixels
            mask = instance_mask == instance_id
            colored[mask] = color
        
        return colored
    
    def _get_instance_color(self, instance_id: int) -> Tuple[int, int, int]:
        """
        Get color for a specific instance ID
        """
        if self._instance_colors is None:
            self._instance_colors = self._generate_instance_colors()
        
        # Use modulo to cycle through colors if we have more instances than colors
        color_idx = (instance_id - 1) % len(self._instance_colors)
        return self._instance_colors[color_idx]
    
    def _generate_instance_colors(self, max_instances: int = 100) -> List[Tuple[int, int, int]]:
        """
        Generate distinct colors for instances
        """
        colors = []
        
        # Generate colors using matplotlib colormap
        cmap = plt.cm.get_cmap(self.colormap)
        for i in range(max_instances):
            color = cmap(i / max_instances)[:3]  # RGB only
            color = tuple(int(c * 255) for c in color)
            colors.append(color)
        
        return colors
    
    def _get_class_colors(self) -> Dict[int, Tuple[int, int, int]]:
        """
        Get predefined colors for different classes
        """
        return {
            0: (0, 0, 0),        # road - black
            1: (128, 128, 128),  # sidewalk - gray
            2: (70, 70, 70),     # building - dark gray
            3: (153, 153, 153),  # wall - light gray
            4: (190, 190, 190),  # fence - silver
            5: (220, 20, 60),    # pole - crimson
            6: (255, 0, 0),      # traffic_light - red
            7: (255, 255, 0),    # traffic_sign - yellow
            8: (0, 255, 0),      # vegetation - green
            9: (107, 142, 35),   # terrain - olive
            10: (135, 206, 235), # sky - sky blue
            11: (255, 20, 147),  # person - deep pink
            12: (255, 105, 180), # rider - hot pink
            13: (0, 0, 255),     # car - blue
            14: (255, 165, 0),   # truck - orange
            15: (255, 0, 255),   # bus - magenta
            16: (0, 255, 255),   # train - cyan
            17: (128, 0, 128),   # motorcycle - purple
            18: (255, 192, 203)  # bicycle - pink
        }
    
    def _add_contours(self, image: np.ndarray, instance_mask: np.ndarray) -> np.ndarray:
        """
        Add contours around instances
        """
        contour_image = image.copy()
        
        # Find contours
        contours, _ = cv2.findContours(
            instance_mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Draw contours
        cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 2)
        
        return contour_image
    
    def _add_labels(
        self, 
        image: np.ndarray, 
        instance_mask: np.ndarray, 
        class_mapping: Dict[int, int]
    ) -> np.ndarray:
        """
        Add instance labels to image
        """
        labeled_image = image.copy()
        
        # Get unique instance IDs
        unique_instances = np.unique(instance_mask)
        
        for instance_id in unique_instances:
            if instance_id == 0:  # Skip background
                continue
            
            # Find centroid of instance
            mask = instance_mask == instance_id
            if np.sum(mask) == 0:
                continue
            
            moments = cv2.moments(mask.astype(np.uint8))
            if moments['m00'] != 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
                
                # Get class ID
                class_id = class_mapping.get(instance_id, 0)
                
                # Add label
                label = f"ID:{instance_id} C:{class_id}"
                cv2.putText(
                    labeled_image, 
                    label, 
                    (cx - 20, cy), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (255, 255, 255), 
                    2
                )
        
        return labeled_image
    
    def create_comparison_plot(
        self,
        original_image: np.ndarray,
        semantic_mask: np.ndarray,
        instances: Dict[str, np.ndarray],
        output_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Create comparison plot showing original, semantic, and instance segmentation
        
        Args:
            original_image: Original RGB image
            semantic_mask: Semantic segmentation mask
            instances: Instance extraction results
            output_path: Path to save plot
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Semantic segmentation
        semantic_colored = self._colorize_semantic_mask(semantic_mask)
        axes[1].imshow(semantic_colored)
        axes[1].set_title('Semantic Segmentation')
        axes[1].axis('off')
        
        # Instance segmentation
        instance_vis = self._create_mask_visualization(
            instances['instances'], 
            instances['class_mapping']
        )
        axes[2].imshow(instance_vis)
        axes[2].set_title('Instance Segmentation')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _colorize_semantic_mask(self, semantic_mask: np.ndarray) -> np.ndarray:
        """
        Colorize semantic segmentation mask
        """
        h, w = semantic_mask.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        
        unique_classes = np.unique(semantic_mask)
        
        for class_id in unique_classes:
            if class_id == 0:  # Skip background
                continue
            
            color = self._class_colors.get(class_id, (128, 128, 128))
            mask = semantic_mask == class_id
            colored[mask] = color
        
        return colored
