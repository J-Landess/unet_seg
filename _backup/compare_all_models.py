#!/usr/bin/env python3
"""
Compare all model results: dummy, trained, and pre-trained models
"""

import os
import cv2
import numpy as np
from pathlib import Path

def compare_all_models():
    """Compare results from all model types"""
    
    print("🔍 Comprehensive Model Comparison")
    print("=" * 50)
    
    # Define model directories
    models = {
        "Dummy Segmentation": "video_inference_output",
        "Trained Model (10 epochs)": "trained_model_inference_output", 
        "Pre-trained VGG11": "pretrained_vgg11_output",
        "Pre-trained EfficientNet": "pretrained_efficientnet_output"
    }
    
    # Check which models are available
    available_models = {}
    for name, dir_path in models.items():
        if os.path.exists(dir_path):
            available_models[name] = dir_path
            print(f"✅ {name}: {dir_path}")
        else:
            print(f"❌ {name}: {dir_path} (not found)")
    
    if len(available_models) < 2:
        print("\n❌ Need at least 2 models to compare!")
        return
    
    print(f"\n📊 Found {len(available_models)} models to compare")
    
    # Find common frames
    all_frames = set()
    for name, dir_path in available_models.items():
        frames = [f for f in os.listdir(dir_path) if f.endswith('_original.jpg')]
        all_frames.update(frames)
    
    common_frames = sorted(list(all_frames))[:3]  # Compare first 3 frames
    
    print(f"\n🎯 Comparing {len(common_frames)} frames across models")
    print("-" * 60)
    
    # Compare each frame
    for frame_file in common_frames:
        base_name = frame_file.replace('_original.jpg', '')
        print(f"\n📸 Frame: {base_name}")
        print("=" * 30)
        
        for model_name, dir_path in available_models.items():
            # Check if this frame exists for this model
            orig_path = os.path.join(dir_path, f"{base_name}_original.jpg")
            mask_path = os.path.join(dir_path, f"{base_name}_mask.png")
            vis_path = os.path.join(dir_path, f"{base_name}_visualization.jpg")
            
            if os.path.exists(orig_path):
                # File sizes
                orig_size = os.path.getsize(orig_path)
                mask_size = os.path.getsize(mask_path) if os.path.exists(mask_path) else 0
                vis_size = os.path.getsize(vis_path) if os.path.exists(vis_path) else 0
                
                # Analyze mask content
                if os.path.exists(mask_path):
                    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask_img is not None:
                        unique_classes = len(np.unique(mask_img))
                        mean_value = np.mean(mask_img)
                        std_value = np.std(mask_img)
                        
                        print(f"  {model_name}:")
                        print(f"    Mask size: {mask_size:,} bytes")
                        print(f"    Classes used: {unique_classes}")
                        print(f"    Mean pixel: {mean_value:.2f} ± {std_value:.2f}")
                    else:
                        print(f"  {model_name}: Could not read mask")
                else:
                    print(f"  {model_name}: No mask found")
            else:
                print(f"  {model_name}: Frame not found")
    
    # Model characteristics summary
    print(f"\n📈 Model Characteristics Summary")
    print("=" * 40)
    
    model_info = {
        "Dummy Segmentation": {
            "Parameters": "N/A",
            "Training": "None (geometric patterns)",
            "Speed": "Fastest",
            "Accuracy": "Low (demo only)",
            "Use Case": "Testing pipeline"
        },
        "Trained Model (10 epochs)": {
            "Parameters": "31M",
            "Training": "10 epochs on dummy data",
            "Speed": "Medium",
            "Accuracy": "Medium",
            "Use Case": "Custom training"
        },
        "Pre-trained VGG11": {
            "Parameters": "18M",
            "Training": "ImageNet + transfer learning",
            "Speed": "Medium",
            "Accuracy": "High",
            "Use Case": "General purpose"
        },
        "Pre-trained EfficientNet": {
            "Parameters": "6M",
            "Training": "ImageNet + transfer learning", 
            "Speed": "Fastest (pre-trained)",
            "Accuracy": "High",
            "Use Case": "Efficient inference"
        }
    }
    
    for model_name in available_models.keys():
        if model_name in model_info:
            info = model_info[model_name]
            print(f"\n{model_name}:")
            for key, value in info.items():
                print(f"  {key}: {value}")
    
    # Recommendations
    print(f"\n💡 Recommendations")
    print("=" * 20)
    print("• For quick testing: Use dummy segmentation")
    print("• For custom datasets: Train your own model")
    print("• For general use: Use pre-trained VGG11")
    print("• For efficiency: Use pre-trained EfficientNet")
    print("• For production: Fine-tune pre-trained models on your data")
    
    # Performance comparison
    print(f"\n⚡ Performance Comparison")
    print("=" * 30)
    
    # Calculate average file sizes
    for model_name, dir_path in available_models.items():
        mask_files = [f for f in os.listdir(dir_path) if f.endswith('_mask.png')]
        if mask_files:
            total_size = sum(os.path.getsize(os.path.join(dir_path, f)) for f in mask_files)
            avg_size = total_size / len(mask_files)
            print(f"{model_name}: {avg_size:,.0f} bytes per mask (avg)")
    
    print(f"\n✅ Comparison complete!")
    print(f"View results in:")
    for name, dir_path in available_models.items():
        print(f"  • {name}: {dir_path}/")

def analyze_segmentation_quality():
    """Analyze the quality of segmentation results"""
    
    print(f"\n🔬 Segmentation Quality Analysis")
    print("=" * 40)
    
    models = {
        "Trained Model": "trained_model_inference_output",
        "VGG11": "pretrained_vgg11_output", 
        "EfficientNet": "pretrained_efficientnet_output"
    }
    
    for model_name, dir_path in models.items():
        if not os.path.exists(dir_path):
            continue
            
        print(f"\n{model_name}:")
        
        # Analyze mask diversity
        mask_files = [f for f in os.listdir(dir_path) if f.endswith('_mask.png')]
        
        if mask_files:
            all_classes = set()
            total_pixels = 0
            class_counts = {}
            
            for mask_file in mask_files[:3]:  # Analyze first 3 masks
                mask_path = os.path.join(dir_path, mask_file)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                
                if mask is not None:
                    unique_classes = np.unique(mask)
                    all_classes.update(unique_classes)
                    total_pixels += mask.size
                    
                    for class_id in unique_classes:
                        count = np.sum(mask == class_id)
                        class_counts[class_id] = class_counts.get(class_id, 0) + count
            
            print(f"  Classes detected: {len(all_classes)}")
            print(f"  Class distribution: {dict(sorted(class_counts.items()))}")
            
            # Calculate class balance
            if class_counts:
                max_count = max(class_counts.values())
                min_count = min(class_counts.values())
                balance_ratio = min_count / max_count if max_count > 0 else 0
                print(f"  Class balance ratio: {balance_ratio:.3f} (1.0 = perfect balance)")

if __name__ == "__main__":
    compare_all_models()
    analyze_segmentation_quality()
