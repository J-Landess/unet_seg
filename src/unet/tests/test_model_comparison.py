#!/usr/bin/env python3
"""
Model comparison testing module
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional

class ModelComparisonTester:
    """Compare different model results"""
    
    def __init__(self, test_outputs_dir: str = "test_outputs"):
        self.test_outputs_dir = Path(test_outputs_dir)
    
    def compare_all_models(self) -> Dict[str, Any]:
        """Compare all available model results"""
        
        print("🔍 Comprehensive Model Comparison")
        print("=" * 50)
        
        # Define model directories
        models = {
            "Dummy Segmentation": "dummy_inference",
            "Trained Model (10 epochs)": "trained_inference", 
            "Pre-trained VGG11": "pretrained_inference",
            "Pre-trained EfficientNet": "pretrained_inference_efficientnet"
        }
        
        # Check which models are available
        available_models = {}
        for name, dir_name in models.items():
            dir_path = self.test_outputs_dir / dir_name
            if dir_path.exists():
                available_models[name] = str(dir_path)
                print(f"✅ {name}: {dir_path}")
            else:
                print(f"❌ {name}: {dir_path} (not found)")
        
        if len(available_models) < 2:
            print("\n❌ Need at least 2 models to compare!")
            return {"error": "Insufficient models for comparison"}
        
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
        frame_comparisons = {}
        for frame_file in common_frames:
            base_name = frame_file.replace('_original.jpg', '')
            frame_comparisons[base_name] = self._compare_frame(
                base_name, available_models
            )
        
        # Generate summary
        summary = self._generate_comparison_summary(available_models, frame_comparisons)
        
        return {
            "available_models": available_models,
            "frame_comparisons": frame_comparisons,
            "summary": summary
        }
    
    def _compare_frame(self, base_name: str, available_models: Dict[str, str]) -> Dict[str, Any]:
        """Compare a single frame across models"""
        
        print(f"\n📸 Frame: {base_name}")
        print("=" * 30)
        
        frame_results = {}
        
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
                mask_analysis = {}
                if os.path.exists(mask_path):
                    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask_img is not None:
                        unique_classes = len(np.unique(mask_img))
                        mean_value = np.mean(mask_img)
                        std_value = np.std(mask_img)
                        
                        mask_analysis = {
                            "unique_classes": unique_classes,
                            "mean_pixel": mean_value,
                            "std_pixel": std_value,
                            "mask_size_bytes": mask_size
                        }
                        
                        print(f"  {model_name}:")
                        print(f"    Mask size: {mask_size:,} bytes")
                        print(f"    Classes used: {unique_classes}")
                        print(f"    Mean pixel: {mean_value:.2f} ± {std_value:.2f}")
                    else:
                        print(f"  {model_name}: Could not read mask")
                        mask_analysis = {"error": "Could not read mask"}
                else:
                    print(f"  {model_name}: No mask found")
                    mask_analysis = {"error": "No mask found"}
                
                frame_results[model_name] = {
                    "original_size_bytes": orig_size,
                    "visualization_size_bytes": vis_size,
                    "mask_analysis": mask_analysis
                }
            else:
                print(f"  {model_name}: Frame not found")
                frame_results[model_name] = {"error": "Frame not found"}
        
        return frame_results
    
    def _generate_comparison_summary(self, available_models: Dict[str, str], frame_comparisons: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comparison summary"""
        
        # Model characteristics
        model_info = {
            "Dummy Segmentation": {
                "parameters": "N/A",
                "training": "None (geometric patterns)",
                "speed": "Fastest",
                "accuracy": "Low (demo only)",
                "use_case": "Testing pipeline"
            },
            "Trained Model (10 epochs)": {
                "parameters": "31M",
                "training": "10 epochs on dummy data",
                "speed": "Medium",
                "accuracy": "Medium",
                "use_case": "Custom training"
            },
            "Pre-trained VGG11": {
                "parameters": "18M",
                "training": "ImageNet + transfer learning",
                "speed": "Medium",
                "accuracy": "High",
                "use_case": "General purpose"
            },
            "Pre-trained EfficientNet": {
                "parameters": "6M",
                "training": "ImageNet + transfer learning", 
                "speed": "Fastest (pre-trained)",
                "accuracy": "High",
                "use_case": "Efficient inference"
            }
        }
        
        # Calculate performance metrics
        performance_metrics = {}
        for model_name in available_models.keys():
            if model_name in model_info:
                # Calculate average mask size
                mask_sizes = []
                for frame_data in frame_comparisons.values():
                    if model_name in frame_data and "mask_analysis" in frame_data[model_name]:
                        mask_size = frame_data[model_name]["mask_analysis"].get("mask_size_bytes", 0)
                        if mask_size > 0:
                            mask_sizes.append(mask_size)
                
                avg_mask_size = np.mean(mask_sizes) if mask_sizes else 0
                
                performance_metrics[model_name] = {
                    **model_info[model_name],
                    "average_mask_size_bytes": avg_mask_size
                }
        
        # Recommendations
        recommendations = [
            "For quick testing: Use dummy segmentation",
            "For custom datasets: Train your own model", 
            "For general use: Use pre-trained VGG11",
            "For efficiency: Use pre-trained EfficientNet",
            "For production: Fine-tune pre-trained models on your data"
        ]
        
        return {
            "model_characteristics": performance_metrics,
            "recommendations": recommendations,
            "total_models_compared": len(available_models)
        }
    
    def analyze_segmentation_quality(self) -> Dict[str, Any]:
        """Analyze segmentation quality across models"""
        
        print(f"\n🔬 Segmentation Quality Analysis")
        print("=" * 40)
        
        models = {
            "Trained Model": "trained_inference",
            "VGG11": "pretrained_inference", 
            "EfficientNet": "pretrained_inference_efficientnet"
        }
        
        quality_analysis = {}
        
        for model_name, dir_name in models.items():
            dir_path = self.test_outputs_dir / dir_name
            if not dir_path.exists():
                continue
                
            print(f"\n{model_name}:")
            
            # Analyze mask diversity
            mask_files = [f for f in os.listdir(dir_path) if f.endswith('_mask.png')]
            
            if mask_files:
                all_classes = set()
                class_counts = {}
                
                for mask_file in mask_files[:3]:  # Analyze first 3 masks
                    mask_path = dir_path / mask_file
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    
                    if mask is not None:
                        unique_classes = np.unique(mask)
                        all_classes.update(unique_classes)
                        
                        for class_id in unique_classes:
                            count = np.sum(mask == class_id)
                            class_counts[class_id] = class_counts.get(class_id, 0) + count
                
                # Calculate class balance
                balance_ratio = 0
                if class_counts:
                    max_count = max(class_counts.values())
                    min_count = min(class_counts.values())
                    balance_ratio = min_count / max_count if max_count > 0 else 0
                
                quality_analysis[model_name] = {
                    "classes_detected": len(all_classes),
                    "class_distribution": dict(sorted(class_counts.items())),
                    "class_balance_ratio": balance_ratio,
                    "masks_analyzed": len(mask_files)
                }
                
                print(f"  Classes detected: {len(all_classes)}")
                print(f"  Class balance ratio: {balance_ratio:.3f} (1.0 = perfect balance)")
        
        return quality_analysis

def main():
    """Example usage"""
    tester = ModelComparisonTester()
    
    # Compare all models
    comparison_results = tester.compare_all_models()
    
    # Analyze quality
    quality_results = tester.analyze_segmentation_quality()
    
    print(f"\n✅ Analysis complete!")
    print(f"View detailed results in: {tester.test_outputs_dir}/")

if __name__ == "__main__":
    main()
