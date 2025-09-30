"""
Visualization utilities for semantic segmentation
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import cv2
from PIL import Image
import torch
from typing import List, Tuple, Optional, Dict, Any
import os


class SegmentationVisualizer:
    """
    Visualization utilities for semantic segmentation results
    """
    
    def __init__(self, class_names: Optional[List[str]] = None, class_colors: Optional[List[List[int]]] = None):
        """
        Initialize visualizer
        
        Args:
            class_names: List of class names
            class_colors: List of RGB colors for each class
        """
        self.class_names = class_names or [f"Class {i}" for i in range(21)]
        self.class_colors = class_colors or self._get_default_colors()
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def _get_default_colors(self) -> List[List[int]]:
        """Get default Pascal VOC colors"""
        return [
            [0, 0, 0],        # background
            [128, 0, 0],      # aeroplane
            [0, 128, 0],      # bicycle
            [128, 128, 0],    # bird
            [0, 0, 128],      # boat
            [128, 0, 128],    # bottle
            [0, 128, 128],    # bus
            [128, 128, 128],  # car
            [64, 0, 0],       # cat
            [192, 0, 0],      # chair
            [64, 128, 0],     # cow
            [192, 128, 0],    # dining table
            [64, 0, 128],     # dog
            [192, 0, 128],    # horse
            [64, 128, 128],   # motorbike
            [192, 128, 128],  # person
            [0, 64, 0],       # potted plant
            [128, 64, 0],     # sheep
            [0, 192, 0],      # sofa
            [128, 192, 0],    # train
            [0, 64, 128]      # tv/monitor
        ]
    
    def create_colored_mask(self, mask: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """
        Create colored mask from segmentation mask
        
        Args:
            mask: Segmentation mask (H, W)
            alpha: Transparency factor
            
        Returns:
            Colored mask (H, W, 3)
        """
        colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        
        for class_id in range(len(self.class_colors)):
            class_mask = mask == class_id
            colored_mask[class_mask] = self.class_colors[class_id]
        
        return colored_mask
    
    def visualize_prediction(
        self, 
        image: np.ndarray, 
        prediction: np.ndarray, 
        ground_truth: Optional[np.ndarray] = None,
        title: str = "Segmentation Results",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create comprehensive visualization of segmentation results
        
        Args:
            image: Original image (H, W, 3)
            prediction: Predicted segmentation mask (H, W)
            ground_truth: Ground truth mask (H, W) - optional
            title: Plot title
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        num_plots = 3 if ground_truth is not None else 2
        fig, axes = plt.subplots(1, num_plots, figsize=(5*num_plots, 5))
        
        if num_plots == 2:
            axes = [axes[0], axes[1]]
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Prediction
        colored_pred = self.create_colored_mask(prediction)
        axes[1].imshow(colored_pred)
        axes[1].set_title("Prediction")
        axes[1].axis('off')
        
        # Ground truth (if provided)
        if ground_truth is not None:
            colored_gt = self.create_colored_mask(ground_truth)
            axes[2].imshow(colored_gt)
            axes[2].set_title("Ground Truth")
            axes[2].axis('off')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_overlay(
        self, 
        image: np.ndarray, 
        mask: np.ndarray, 
        alpha: float = 0.6,
        title: str = "Segmentation Overlay",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create overlay visualization
        
        Args:
            image: Original image (H, W, 3)
            mask: Segmentation mask (H, W)
            alpha: Transparency for overlay
            title: Plot title
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        # Show original image
        ax.imshow(image)
        
        # Create colored mask
        colored_mask = self.create_colored_mask(mask)
        
        # Overlay mask
        ax.imshow(colored_mask, alpha=alpha)
        
        ax.set_title(title)
        ax.axis('off')
        
        # Add legend
        self._add_legend(ax)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _add_legend(self, ax: plt.Axes):
        """Add legend to plot"""
        legend_elements = []
        for i, (name, color) in enumerate(zip(self.class_names, self.class_colors)):
            if i < len(self.class_names):  # Only show classes that exist
                color_normalized = [c/255.0 for c in color]
                legend_elements.append(
                    patches.Patch(color=color_normalized, label=name)
                )
        
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    def plot_training_curves(
        self, 
        train_losses: List[float], 
        val_losses: List[float],
        train_metrics: List[Dict[str, float]],
        val_metrics: List[Dict[str, float]],
        metric_name: str = 'mean_iou',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot training curves
        
        Args:
            train_losses: List of training losses
            val_losses: List of validation losses
            train_metrics: List of training metrics dictionaries
            val_metrics: List of validation metrics dictionaries
            metric_name: Name of metric to plot
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = range(1, len(train_losses) + 1)
        
        # Plot losses
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot metrics
        train_metric_values = [m.get(metric_name, 0) for m in train_metrics]
        val_metric_values = [m.get(metric_name, 0) for m in val_metrics]
        
        ax2.plot(epochs, train_metric_values, 'b-', label=f'Training {metric_name}')
        ax2.plot(epochs, val_metric_values, 'r-', label=f'Validation {metric_name}')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel(metric_name.replace('_', ' ').title())
        ax2.set_title(f'Training and Validation {metric_name.replace("_", " ").title()}')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_confusion_matrix(
        self, 
        confusion_matrix: np.ndarray,
        class_names: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot confusion matrix
        
        Args:
            confusion_matrix: Confusion matrix (num_classes, num_classes)
            class_names: List of class names
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        if class_names is None:
            class_names = self.class_names
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Normalize confusion matrix
        cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax
        )
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Normalized Confusion Matrix')
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_class_distribution(
        self, 
        masks: List[np.ndarray],
        title: str = "Class Distribution",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot class distribution in dataset
        
        Args:
            masks: List of segmentation masks
            title: Plot title
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        # Count pixels for each class
        class_counts = np.zeros(len(self.class_names))
        
        for mask in masks:
            unique, counts = np.unique(mask, return_counts=True)
            for class_id, count in zip(unique, counts):
                if class_id < len(class_counts):
                    class_counts[class_id] += count
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars = ax.bar(range(len(self.class_names)), class_counts)
        ax.set_xlabel('Class')
        ax.set_ylabel('Number of Pixels')
        ax.set_title(title)
        ax.set_xticks(range(len(self.class_names)))
        ax.set_xticklabels(self.class_names, rotation=45, ha='right')
        
        # Color bars according to class colors
        for i, (bar, color) in enumerate(zip(bars, self.class_colors)):
            if i < len(self.class_colors):
                color_normalized = [c/255.0 for c in color]
                bar.set_color(color_normalized)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_comparison_grid(
        self, 
        images: List[np.ndarray],
        predictions: List[np.ndarray],
        ground_truths: Optional[List[np.ndarray]] = None,
        titles: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create grid comparison of multiple predictions
        
        Args:
            images: List of original images
            predictions: List of predicted masks
            ground_truths: List of ground truth masks (optional)
            titles: List of titles for each sample
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        num_samples = len(images)
        num_cols = 3 if ground_truths is not None else 2
        num_rows = num_samples
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 5*num_rows))
        
        if num_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_samples):
            # Original image
            axes[i, 0].imshow(images[i])
            axes[i, 0].set_title(f"{titles[i] if titles else f'Sample {i+1}'} - Original")
            axes[i, 0].axis('off')
            
            # Prediction
            colored_pred = self.create_colored_mask(predictions[i])
            axes[i, 1].imshow(colored_pred)
            axes[i, 1].set_title(f"{titles[i] if titles else f'Sample {i+1}'} - Prediction")
            axes[i, 1].axis('off')
            
            # Ground truth (if provided)
            if ground_truths is not None:
                colored_gt = self.create_colored_mask(ground_truths[i])
                axes[i, 2].imshow(colored_gt)
                axes[i, 2].set_title(f"{titles[i] if titles else f'Sample {i+1}'} - Ground Truth")
                axes[i, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def save_prediction_visualization(
    image_path: str,
    prediction: np.ndarray,
    output_path: str,
    class_colors: Optional[List[List[int]]] = None,
    alpha: float = 0.6
):
    """
    Quick function to save prediction visualization
    
    Args:
        image_path: Path to original image
        prediction: Segmentation prediction
        output_path: Path to save visualization
        class_colors: Class colors
        alpha: Overlay transparency
    """
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create visualizer
    visualizer = SegmentationVisualizer(class_colors=class_colors)
    
    # Create overlay
    fig = visualizer.visualize_overlay(image, prediction, alpha=alpha)
    
    # Save
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    # Example usage
    visualizer = SegmentationVisualizer()
    
    # Create dummy data
    image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    prediction = np.random.randint(0, 21, (256, 256))
    
    # Create visualization
    fig = visualizer.visualize_prediction(image, prediction)
    plt.show()
