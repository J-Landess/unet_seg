#!/usr/bin/env python3
"""
Compare dummy segmentation vs trained model results
"""

import os
import cv2
import numpy as np
from pathlib import Path

def compare_results():
    """Compare the results from dummy vs trained model inference"""
    
    print("🔍 Comparing Inference Results")
    print("=" * 40)
    
    dummy_dir = "video_inference_output"
    trained_dir = "trained_model_inference_output"
    
    if not os.path.exists(dummy_dir) or not os.path.exists(trained_dir):
        print("❌ One or both output directories not found!")
        return
    
    # Find common frames
    dummy_frames = [f for f in os.listdir(dummy_dir) if f.endswith('_original.jpg')]
    trained_frames = [f for f in os.listdir(trained_dir) if f.endswith('_original.jpg')]
    
    print(f"📊 Dummy inference frames: {len(dummy_frames)}")
    print(f"📊 Trained model frames: {len(trained_frames)}")
    
    # Compare file sizes and properties
    print("\n📈 File Size Comparison:")
    print("-" * 30)
    
    for frame_file in sorted(dummy_frames)[:3]:  # Show first 3 frames
        base_name = frame_file.replace('_original.jpg', '')
        
        dummy_orig = os.path.join(dummy_dir, f"{base_name}_original.jpg")
        dummy_vis = os.path.join(dummy_dir, f"{base_name}_visualization.jpg")
        dummy_mask = os.path.join(dummy_dir, f"{base_name}_mask.png")
        
        trained_orig = os.path.join(trained_dir, f"{base_name}_original.jpg")
        trained_vis = os.path.join(trained_dir, f"{base_name}_visualization.jpg")
        trained_mask = os.path.join(trained_dir, f"{base_name}_mask.png")
        
        if os.path.exists(trained_orig):
            print(f"\nFrame {base_name}:")
            
            # File sizes
            dummy_orig_size = os.path.getsize(dummy_orig) if os.path.exists(dummy_orig) else 0
            trained_orig_size = os.path.getsize(trained_orig) if os.path.exists(trained_orig) else 0
            
            dummy_vis_size = os.path.getsize(dummy_vis) if os.path.exists(dummy_vis) else 0
            trained_vis_size = os.path.getsize(trained_vis) if os.path.exists(trained_vis) else 0
            
            dummy_mask_size = os.path.getsize(dummy_mask) if os.path.exists(dummy_mask) else 0
            trained_mask_size = os.path.getsize(trained_mask) if os.path.exists(trained_mask) else 0
            
            print(f"  Original: {dummy_orig_size:,} vs {trained_orig_size:,} bytes")
            print(f"  Visualization: {dummy_vis_size:,} vs {trained_vis_size:,} bytes")
            print(f"  Mask: {dummy_mask_size:,} vs {trained_mask_size:,} bytes")
            
            # Analyze mask content
            if os.path.exists(dummy_mask) and os.path.exists(trained_mask):
                dummy_mask_img = cv2.imread(dummy_mask, cv2.IMREAD_GRAYSCALE)
                trained_mask_img = cv2.imread(trained_mask, cv2.IMREAD_GRAYSCALE)
                
                if dummy_mask_img is not None and trained_mask_img is not None:
                    dummy_unique = len(np.unique(dummy_mask_img))
                    trained_unique = len(np.unique(trained_mask_img))
                    
                    print(f"  Unique classes: {dummy_unique} vs {trained_unique}")
                    
                    # Calculate some basic statistics
                    dummy_mean = np.mean(dummy_mask_img)
                    trained_mean = np.mean(trained_mask_img)
                    
                    print(f"  Mean pixel value: {dummy_mean:.2f} vs {trained_mean:.2f}")
    
    print("\n🎯 Key Differences:")
    print("-" * 20)
    print("• Dummy segmentation: Simple geometric patterns based on image intensity")
    print("• Trained model: Learned features from 10 epochs of training")
    print("• Trained model should show more realistic segmentation patterns")
    print("• Both use the same 21-class color scheme for visualization")
    
    print("\n📁 Output Locations:")
    print(f"• Dummy results: {dummy_dir}/")
    print(f"• Trained results: {trained_dir}/")
    
    print("\n✅ Comparison complete!")

if __name__ == "__main__":
    compare_results()
