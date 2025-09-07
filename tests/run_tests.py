#!/usr/bin/env python3
"""
Main test runner for U-Net semantic segmentation project
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from tests.test_video_inference import VideoInferenceTester
from tests.test_model_comparison import ModelComparisonTester
from tests.test_training import TrainingTester

def run_video_inference_tests(video_path: str, max_frames: int = 5):
    """Run video inference tests"""
    print("🎬 Running Video Inference Tests")
    print("=" * 50)
    
    tester = VideoInferenceTester()
    
    # Test dummy inference
    print("\n1. Testing Dummy Segmentation...")
    dummy_results = tester.test_dummy_inference(video_path, max_frames=max_frames)
    
    # Test trained model (if available)
    if Path("checkpoints/best_model.pth").exists():
        print("\n2. Testing Trained Model...")
        trained_results = tester.test_trained_model_inference(video_path, max_frames=max_frames)
    else:
        print("\n2. Skipping Trained Model (no model found)")
        trained_results = None
    
    # Test pre-trained models
    print("\n3. Testing Pre-trained VGG11...")
    vgg11_results = tester.test_pretrained_inference(
        video_path, encoder_name="vgg11", max_frames=max_frames
    )
    
    print("\n4. Testing Pre-trained EfficientNet...")
    efficientnet_results = tester.test_pretrained_inference(
        video_path, encoder_name="efficientnet", max_frames=max_frames
    )
    
    return {
        "dummy": dummy_results,
        "trained": trained_results,
        "vgg11": vgg11_results,
        "efficientnet": efficientnet_results
    }

def run_model_comparison_tests():
    """Run model comparison tests"""
    print("\n🔍 Running Model Comparison Tests")
    print("=" * 50)
    
    tester = ModelComparisonTester()
    
    # Compare all models
    comparison_results = tester.compare_all_models()
    
    # Analyze quality
    quality_results = tester.analyze_segmentation_quality()
    
    return {
        "comparison": comparison_results,
        "quality": quality_results
    }

def run_training_tests(num_epochs: int = 3):
    """Run training tests"""
    print("\n🚀 Running Training Tests")
    print("=" * 50)
    
    tester = TrainingTester()
    
    # Test simple training
    print("\n1. Testing Simple Training...")
    simple_results = tester.test_simple_training(num_epochs=num_epochs)
    
    # Test pre-trained training
    print("\n2. Testing Pre-trained Training...")
    pretrained_results = tester.test_pretrained_training(
        encoder_name="vgg11", num_epochs=num_epochs
    )
    
    return {
        "simple": simple_results,
        "pretrained": pretrained_results
    }

def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(description="Run U-Net tests")
    parser.add_argument("--video", default="video_processing/data3_users_yoavnavon_clips_dqa_glued_sv_vehicle_day_FRONT_RIGHT_GENERIC_1x.mp4", help="Video path for testing")
    parser.add_argument("--max-frames", type=int, default=5, help="Maximum frames to process")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--tests", nargs="+", choices=["inference", "comparison", "training", "all"], default=["all"], help="Tests to run")
    
    args = parser.parse_args()
    
    print("🧪 U-Net Semantic Segmentation Test Suite")
    print("=" * 60)
    print(f"Video: {args.video}")
    print(f"Max frames: {args.max_frames}")
    print(f"Training epochs: {args.epochs}")
    print(f"Tests: {', '.join(args.tests)}")
    print()
    
    results = {}
    
    # Run selected tests
    if "inference" in args.tests or "all" in args.tests:
        results["inference"] = run_video_inference_tests(args.video, args.max_frames)
    
    if "comparison" in args.tests or "all" in args.tests:
        results["comparison"] = run_model_comparison_tests()
    
    if "training" in args.tests or "all" in args.tests:
        results["training"] = run_training_tests(args.epochs)
    
    # Summary
    print("\n🎉 Test Suite Completed!")
    print("=" * 40)
    print("Results saved to:")
    print("  • test_outputs/ - All test outputs")
    print("  • checkpoints/ - Trained models")
    print("  • logs/ - Training logs")
    
    return results

if __name__ == "__main__":
    main()
