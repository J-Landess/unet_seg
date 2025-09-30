"""
Inference module for semantic segmentation
"""

import os
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import List, Tuple, Optional, Union
import yaml

from ..models import create_unet_model


class SegmentationInference:
    """
    Inference class for semantic segmentation models
    """
    
    def __init__(self, model_path: str, config_path: str, device: Optional[str] = None):
        """
        Initialize inference engine
        
        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to configuration file
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
        """
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Setup preprocessing
        self.transform = self._get_transform()
        
        # Class colors for visualization
        self.class_colors = self._get_class_colors()
    
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Load trained model from checkpoint"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create model from config
        model = create_unet_model(self.config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        
        return model
    
    def _get_transform(self) -> A.Compose:
        """Get preprocessing transforms"""
        data_config = self.config.get('data', {})
        image_size = tuple(data_config.get('image_size', [512, 512]))
        
        return A.Compose([
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def _get_class_colors(self) -> List[List[int]]:
        """Get class colors for visualization"""
        # Default Pascal VOC colors
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
    
    def preprocess_image(self, image: Union[str, np.ndarray, Image.Image]) -> torch.Tensor:
        """
        Preprocess image for inference
        
        Args:
            image: Input image (path, numpy array, or PIL Image)
            
        Returns:
            Preprocessed image tensor
        """
        # Load image
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image = np.array(image)
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Assume RGB
                pass
            else:
                raise ValueError("Invalid image format")
        else:
            raise ValueError("Unsupported image type")
        
        # Apply transforms
        transformed = self.transform(image=image)
        image_tensor = transformed['image'].unsqueeze(0)  # Add batch dimension
        
        return image_tensor.to(self.device)
    
    def predict_single(self, image: Union[str, np.ndarray, Image.Image]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict segmentation for a single image
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        # Preprocess
        image_tensor = self.preprocess_image(image)
        
        # Inference
        with torch.no_grad():
            logits = self.model(image_tensor)
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
        
        # Convert to numpy
        predictions = predictions.squeeze(0).cpu().numpy()
        probabilities = probabilities.squeeze(0).cpu().numpy()
        
        return predictions, probabilities
    
    def predict_batch(self, images: List[Union[str, np.ndarray, Image.Image]]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Predict segmentation for a batch of images
        
        Args:
            images: List of input images
            
        Returns:
            Tuple of (predictions_list, probabilities_list)
        """
        predictions_list = []
        probabilities_list = []
        
        for image in images:
            pred, prob = self.predict_single(image)
            predictions_list.append(pred)
            probabilities_list.append(prob)
        
        return predictions_list, probabilities_list
    
    def visualize_prediction(
        self, 
        image: Union[str, np.ndarray, Image.Image], 
        prediction: np.ndarray,
        alpha: float = 0.6
    ) -> np.ndarray:
        """
        Create visualization of prediction overlaid on original image
        
        Args:
            image: Original image
            prediction: Segmentation prediction
            alpha: Transparency for overlay
            
        Returns:
            Visualization image
        """
        # Load original image
        if isinstance(image, str):
            original = cv2.imread(image)
            original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            original = np.array(image)
        else:
            original = image.copy()
        
        # Resize original to match prediction size
        if original.shape[:2] != prediction.shape:
            original = cv2.resize(original, (prediction.shape[1], prediction.shape[0]))
        
        # Create colored mask
        colored_mask = np.zeros_like(original)
        for class_id in range(len(self.class_colors)):
            mask = prediction == class_id
            colored_mask[mask] = self.class_colors[class_id]
        
        # Blend images
        visualization = cv2.addWeighted(original, 1-alpha, colored_mask, alpha, 0)
        
        return visualization
    
    def save_prediction(
        self, 
        prediction: np.ndarray, 
        output_path: str, 
        format: str = 'png'
    ):
        """
        Save prediction mask to file
        
        Args:
            prediction: Segmentation prediction
            output_path: Output file path
            format: Output format ('png', 'jpg', 'npy')
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if format == 'png':
            # Save as grayscale PNG
            cv2.imwrite(output_path, prediction.astype(np.uint8))
        elif format == 'jpg':
            # Save as JPEG (may lose some precision)
            cv2.imwrite(output_path, prediction.astype(np.uint8))
        elif format == 'npy':
            # Save as numpy array
            np.save(output_path, prediction)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def evaluate_on_dataset(
        self, 
        dataset_path: str, 
        output_dir: str,
        save_visualizations: bool = True
    ) -> dict:
        """
        Evaluate model on a dataset
        
        Args:
            dataset_path: Path to dataset directory
            output_dir: Directory to save results
            save_visualizations: Whether to save visualization images
            
        Returns:
            Evaluation metrics
        """
        from ..utils import SegmentationMetrics
        
        # Setup
        os.makedirs(output_dir, exist_ok=True)
        metrics_calc = SegmentationMetrics(
            num_classes=self.config.get('model', {}).get('num_classes', 21)
        )
        
        # Get image and mask paths
        image_dir = os.path.join(dataset_path, 'images')
        mask_dir = os.path.join(dataset_path, 'masks')
        
        image_files = [f for f in os.listdir(image_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        print(f"Evaluating on {len(image_files)} images...")
        
        for i, image_file in enumerate(image_files):
            # Load image and ground truth
            image_path = os.path.join(image_dir, image_file)
            mask_name = os.path.splitext(image_file)[0] + '.png'
            mask_path = os.path.join(mask_dir, mask_name)
            
            if not os.path.exists(mask_path):
                print(f"Warning: No mask found for {image_file}")
                continue
            
            # Predict
            prediction, _ = self.predict_single(image_path)
            
            # Load ground truth
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            # Update metrics
            pred_tensor = torch.from_numpy(prediction)
            gt_tensor = torch.from_numpy(gt_mask)
            metrics_calc.update(pred_tensor, gt_tensor)
            
            # Save visualization if requested
            if save_visualizations:
                vis = self.visualize_prediction(image_path, prediction)
                vis_path = os.path.join(output_dir, f'vis_{image_file}')
                cv2.imwrite(vis_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            
            # Save prediction
            pred_path = os.path.join(output_dir, f'pred_{mask_name}')
            self.save_prediction(prediction, pred_path)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(image_files)} images")
        
        # Compute final metrics
        metrics = metrics_calc.compute()
        
        # Save metrics
        import json
        metrics_path = os.path.join(output_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Evaluation completed. Metrics saved to {metrics_path}")
        return metrics


if __name__ == "__main__":
    # Example usage
    inference = SegmentationInference(
        model_path="checkpoints/best_model.pth",
        config_path="config/config.yaml"
    )
    
    # Predict on single image
    prediction, probabilities = inference.predict_single("test_image.jpg")
    
    # Create visualization
    visualization = inference.visualize_prediction("test_image.jpg", prediction)
    
    # Save results
    inference.save_prediction(prediction, "outputs/prediction.png")
    cv2.imwrite("outputs/visualization.jpg", cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
