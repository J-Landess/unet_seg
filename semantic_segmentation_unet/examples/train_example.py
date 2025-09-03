#!/usr/bin/env python3
"""
Example script for training a semantic segmentation model
"""

import os
import sys
import yaml
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from training import SegmentationTrainer


def main():
    """Example training script"""
    
    # Configuration
    config_path = "config/config.yaml"
    
    # Check if config exists
    if not os.path.exists(config_path):
        print(f"Configuration file not found: {config_path}")
        print("Please create a config file or run 'python main.py setup' first")
        return
    
    # Load and display config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("Training Configuration:")
    print(f"Model: {config.get('model', {}).get('name', 'unet')}")
    print(f"Classes: {config.get('model', {}).get('num_classes', 21)}")
    print(f"Batch Size: {config.get('training', {}).get('batch_size', 8)}")
    print(f"Epochs: {config.get('training', {}).get('num_epochs', 100)}")
    print(f"Learning Rate: {config.get('training', {}).get('learning_rate', 0.001)}")
    
    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = SegmentationTrainer(config_path)
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    print("Training completed!")


if __name__ == "__main__":
    main()
