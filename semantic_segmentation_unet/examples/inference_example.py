#!/usr/bin/env python3
"""
Example script for running inference with a trained model
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from inference import SegmentationInference
from utils import SegmentationVisualizer


def main():
    """Example inference script"""
    
    # Configuration
    model_path = "checkpoints/best_model.pth"
    config_path = "config/config.yaml"
    input_image = "data/test/images/sample.jpg"  # Replace with your image path
    output_dir = "outputs"
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Please train a model first or check the path")
        return
    
    if not os.path.exists(config_path):
        print(f"Configuration file not found: {config_path}")
        return
    
    if not os.path.exists(input_image):
        print(f"Input image not found: {input_image}")
        print("Please provide a valid image path")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Initializing inference engine...")
    inference = SegmentationInference(model_path, config_path)
    
    print(f"Running inference on: {input_image}")
    
    # Run inference
    prediction, probabilities = inference.predict_single(input_image)
    
    print(f"Prediction shape: {prediction.shape}")
    print(f"Probabilities shape: {probabilities.shape}")
    print(f"Unique classes in prediction: {np.unique(prediction)}")
    
    # Create visualization
    print("Creating visualization...")
    visualization = inference.visualize_prediction(input_image, prediction)
    
    # Save results
    base_name = os.path.splitext(os.path.basename(input_image))[0]
    
    # Save prediction mask
    pred_path = os.path.join(output_dir, f"{base_name}_prediction.png")
    inference.save_prediction(prediction, pred_path)
    print(f"Prediction saved to: {pred_path}")
    
    # Save visualization
    vis_path = os.path.join(output_dir, f"{base_name}_visualization.jpg")
    cv2.imwrite(vis_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    print(f"Visualization saved to: {vis_path}")
    
    # Create detailed visualization using the visualizer
    print("Creating detailed visualization...")
    visualizer = SegmentationVisualizer()
    
    # Load original image
    original_image = cv2.imread(input_image)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # Create comprehensive visualization
    fig = visualizer.visualize_prediction(
        image=original_image,
        prediction=prediction,
        title=f"Segmentation Results - {base_name}"
    )
    
    # Save detailed visualization
    detailed_vis_path = os.path.join(output_dir, f"{base_name}_detailed_visualization.png")
    fig.savefig(detailed_vis_path, dpi=300, bbox_inches='tight')
    print(f"Detailed visualization saved to: {detailed_vis_path}")
    
    print("Inference completed!")


if __name__ == "__main__":
    main()
