#!/usr/bin/env python3
"""
Main entry point for semantic segmentation U-Net project
"""

import argparse
import os
import sys
import yaml
import cv2
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from training import SegmentationTrainer
from inference import SegmentationInference
from utils import SegmentationVisualizer


def train_model(config_path: str, resume_from: str = None):
    """
    Train the semantic segmentation model
    
    Args:
        config_path: Path to configuration file
        resume_from: Path to checkpoint to resume from (optional)
    """
    print("Starting training...")
    print(f"Config: {config_path}")
    if resume_from:
        print(f"Resuming from: {resume_from}")
    
    trainer = SegmentationTrainer(config_path)
    trainer.train(resume_from=resume_from)
    
    print("Training completed!")


def run_inference(
    model_path: str, 
    config_path: str, 
    input_path: str, 
    output_dir: str,
    batch_mode: bool = False
):
    """
    Run inference on images
    
    Args:
        model_path: Path to trained model
        config_path: Path to configuration file
        input_path: Path to input image or directory
        output_dir: Directory to save outputs
        batch_mode: Whether to process multiple images
    """
    print("Starting inference...")
    print(f"Model: {model_path}")
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize inference engine
    inference = SegmentationInference(model_path, config_path)
    
    if batch_mode:
        # Process directory of images
        if not os.path.isdir(input_path):
            raise ValueError("Input path must be a directory for batch mode")
        
        image_files = [f for f in os.listdir(input_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
        
        print(f"Processing {len(image_files)} images...")
        
        for i, image_file in enumerate(image_files):
            image_path = os.path.join(input_path, image_file)
            
            # Predict
            prediction, probabilities = inference.predict_single(image_path)
            
            # Create visualization
            visualization = inference.visualize_prediction(image_path, prediction)
            
            # Save results
            base_name = os.path.splitext(image_file)[0]
            
            # Save prediction mask
            pred_path = os.path.join(output_dir, f"{base_name}_prediction.png")
            inference.save_prediction(prediction, pred_path)
            
            # Save visualization
            vis_path = os.path.join(output_dir, f"{base_name}_visualization.jpg")
            cv2.imwrite(vis_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(image_files)} images")
    
    else:
        # Process single image
        if not os.path.isfile(input_path):
            raise ValueError("Input path must be a file for single image mode")
        
        # Predict
        prediction, probabilities = inference.predict_single(input_path)
        
        # Create visualization
        visualization = inference.visualize_prediction(input_path, prediction)
        
        # Save results
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        
        # Save prediction mask
        pred_path = os.path.join(output_dir, f"{base_name}_prediction.png")
        inference.save_prediction(prediction, pred_path)
        
        # Save visualization
        vis_path = os.path.join(output_dir, f"{base_name}_visualization.jpg")
        cv2.imwrite(vis_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    
    print("Inference completed!")


def evaluate_model(
    model_path: str, 
    config_path: str, 
    dataset_path: str, 
    output_dir: str
):
    """
    Evaluate model on a dataset
    
    Args:
        model_path: Path to trained model
        config_path: Path to configuration file
        dataset_path: Path to evaluation dataset
        output_dir: Directory to save evaluation results
    """
    print("Starting evaluation...")
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_dir}")
    
    # Initialize inference engine
    inference = SegmentationInference(model_path, config_path)
    
    # Run evaluation
    metrics = inference.evaluate_on_dataset(
        dataset_path=dataset_path,
        output_dir=output_dir,
        save_visualizations=True
    )
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Pixel Accuracy: {metrics['accuracy']:.4f}")
    print(f"Mean Accuracy: {metrics['mean_accuracy']:.4f}")
    print(f"Mean IoU: {metrics['mean_iou']:.4f}")
    
    # Print per-class IoU
    print("\nPer-class IoU:")
    for i, iou in enumerate(metrics['class_ious']):
        class_name = f"Class {i}"
        print(f"{class_name}: {iou:.4f}")
    
    print("Evaluation completed!")


def create_sample_data():
    """Create sample data structure for testing"""
    print("Creating sample data structure...")
    
    # Create directories
    dirs_to_create = [
        "data/train/images",
        "data/train/masks", 
        "data/val/images",
        "data/val/masks",
        "data/test/images",
        "data/test/masks",
        "checkpoints",
        "logs",
        "outputs"
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")
    
    # Create sample config if it doesn't exist
    config_path = "config/config.yaml"
    if not os.path.exists(config_path):
        print(f"Please create your configuration file at: {config_path}")
        print("You can use the provided config/config.yaml as a template.")
    
    print("Sample data structure created!")
    print("\nNext steps:")
    print("1. Add your training images to data/train/images/")
    print("2. Add corresponding masks to data/train/masks/")
    print("3. Add validation data to data/val/")
    print("4. Update config/config.yaml with your settings")
    print("5. Run training with: python main.py train --config config/config.yaml")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Semantic Segmentation U-Net")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--config', required=True, help='Path to config file')
    train_parser.add_argument('--resume', help='Path to checkpoint to resume from')
    
    # Inference command
    infer_parser = subparsers.add_parser('infer', help='Run inference')
    infer_parser.add_argument('--model', required=True, help='Path to trained model')
    infer_parser.add_argument('--config', required=True, help='Path to config file')
    infer_parser.add_argument('--input', required=True, help='Path to input image or directory')
    infer_parser.add_argument('--output', required=True, help='Output directory')
    infer_parser.add_argument('--batch', action='store_true', help='Process multiple images')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate the model')
    eval_parser.add_argument('--model', required=True, help='Path to trained model')
    eval_parser.add_argument('--config', required=True, help='Path to config file')
    eval_parser.add_argument('--dataset', required=True, help='Path to evaluation dataset')
    eval_parser.add_argument('--output', required=True, help='Output directory')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Create sample data structure')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_model(args.config, args.resume)
    elif args.command == 'infer':
        run_inference(args.model, args.config, args.input, args.output, args.batch)
    elif args.command == 'evaluate':
        evaluate_model(args.model, args.config, args.dataset, args.output)
    elif args.command == 'setup':
        create_sample_data()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
