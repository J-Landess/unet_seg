#!/usr/bin/env python3
"""
Main test runner for U-Net semantic segmentation project

This updated runner works with the new consolidated test structure:
- tests/core_tests.py - Core functionality tests
- tests/inference_tests.py - All inference testing
- tests/training_tests.py - Training and model tests
- examples/demo_scripts.py - Demo scripts
"""

import sys
import argparse
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from tests.core_tests import CoreTester
from tests.inference_tests import InferenceTester
from tests.training_tests import TrainingTester
from examples.demo_scripts import DemoScripts


def run_core_tests(video_path: str = None) -> dict:
    """Run core functionality tests"""
    print("🔧 Running Core Functionality Tests")
    print("=" * 50)
    
    tester = CoreTester()
    results = tester.run_all_tests(video_path)
    
    return results


def run_inference_tests(video_path: str, model_path: str = None) -> dict:
    """Run all inference tests"""
    print("🎬 Running Inference Tests")
    print("=" * 50)
    
    tester = InferenceTester()
    results = tester.run_all_inference_tests(video_path, model_path)
    
    return results


def run_training_tests() -> dict:
    """Run all training tests"""
    print("🚀 Running Training Tests")
    print("=" * 50)
    
    tester = TrainingTester()
    results = tester.run_all_training_tests()
    
    return results


def run_demo_scripts(video_path: str = None) -> dict:
    """Run all demo scripts"""
    print("🎭 Running Demo Scripts")
    print("=" * 50)
    
    demos = DemoScripts()
    results = demos.run_all_demos(video_path)
    
    return results


def run_quick_validation() -> dict:
    """Run quick validation tests (core + basic inference)"""
    print("⚡ Running Quick Validation")
    print("=" * 50)
    
    results = {}
    
    # Core tests
    print("\n1. Core Functionality...")
    core_tester = CoreTester()
    results['core'] = core_tester.run_all_tests()
    
    # Basic inference test
    print("\n2. Basic Inference...")
    inference_tester = InferenceTester()
    
    # Check for video file
    video_path = "video_processing/data3_users_yoavnavon_clips_dqa_glued_sv_vehicle_day_FRONT_RIGHT_GENERIC_1x.mp4"
    if os.path.exists(video_path):
        results['inference'] = inference_tester.test_dummy_inference(video_path, max_frames=3)
    else:
        print("   ⚠️ No video file found, skipping inference tests")
        results['inference'] = {'success': False, 'error': 'No video file found'}
    
    return results


def run_comprehensive_tests(video_path: str, model_path: str = None) -> dict:
    """Run comprehensive test suite"""
    print("🧪 Running Comprehensive Test Suite")
    print("=" * 60)
    
    results = {}
    
    # Core functionality
    print("\n1. Core Functionality Tests...")
    results['core'] = run_core_tests(video_path)
    
    # Inference tests
    print("\n2. Inference Tests...")
    results['inference'] = run_inference_tests(video_path, model_path)
    
    # Training tests
    print("\n3. Training Tests...")
    results['training'] = run_training_tests()
    
    # Demo scripts
    print("\n4. Demo Scripts...")
    results['demos'] = run_demo_scripts(video_path)
    
    return results


def print_test_summary(results: dict):
    """Print a summary of test results"""
    print("\n📋 Test Summary")
    print("=" * 50)
    
    total_tests = 0
    passed_tests = 0
    
    for category, category_results in results.items():
        if isinstance(category_results, dict):
            if 'overall_success' in category_results:
                # Core tests
                status = "✅" if category_results['overall_success'] else "❌"
                print(f"{category.upper()}: {status}")
                total_tests += 1
                if category_results['overall_success']:
                    passed_tests += 1
            elif 'success' in category_results:
                # Single test result
                status = "✅" if category_results['success'] else "❌"
                print(f"{category.upper()}: {status}")
                total_tests += 1
                if category_results['success']:
                    passed_tests += 1
            else:
                # Multiple test results
                print(f"{category.upper()}:")
                for test_name, test_result in category_results.items():
                    if isinstance(test_result, dict) and 'success' in test_result:
                        status = "✅" if test_result['success'] else "❌"
                        print(f"  {test_name}: {status}")
                        total_tests += 1
                        if test_result['success']:
                            passed_tests += 1
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed!")
    else:
        print(f"⚠️ {total_tests - passed_tests} tests failed")


def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(description="Run U-Net tests with consolidated structure")
    parser.add_argument("--video", 
                       default="video_processing/data3_users_yoavnavon_clips_dqa_glued_sv_vehicle_day_FRONT_RIGHT_GENERIC_1x.mp4", 
                       help="Video path for testing")
    parser.add_argument("--model", 
                       default="checkpoints/best_model.pth", 
                       help="Path to trained model")
    parser.add_argument("--tests", 
                       nargs="+", 
                       choices=["core", "inference", "training", "demos", "quick", "all"], 
                       default=["all"], 
                       help="Tests to run")
    parser.add_argument("--max-frames", 
                       type=int, 
                       default=5, 
                       help="Maximum frames to process (for inference tests)")
    
    args = parser.parse_args()
    
    print("🧪 U-Net Semantic Segmentation Test Suite")
    print("=" * 60)
    print(f"Video: {args.video}")
    print(f"Model: {args.model}")
    print(f"Tests: {', '.join(args.tests)}")
    print(f"Max frames: {args.max_frames}")
    print()
    
    # Check if video file exists
    if not os.path.exists(args.video):
        print(f"⚠️ Video file not found: {args.video}")
        print("   Some tests will be skipped or use dummy data")
        args.video = None
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"⚠️ Model file not found: {args.model}")
        print("   Trained model tests will be skipped")
        args.model = None
    
    results = {}
    
    # Run selected tests
    if "core" in args.tests or "all" in args.tests:
        results["core"] = run_core_tests(args.video)
    
    if "inference" in args.tests or "all" in args.tests:
        results["inference"] = run_inference_tests(args.video, args.model)
    
    if "training" in args.tests or "all" in args.tests:
        results["training"] = run_training_tests()
    
    if "demos" in args.tests or "all" in args.tests:
        results["demos"] = run_demo_scripts(args.video)
    
    if "quick" in args.tests:
        results = run_quick_validation()
    
    # Print summary
    print_test_summary(results)
    
    # Print output locations
    print("\n📁 Output Locations")
    print("=" * 30)
    print("• test_outputs/core_tests/ - Core functionality test results")
    print("• test_outputs/inference_tests/ - Inference test results")
    print("• test_outputs/training_tests/ - Training test results")
    print("• demo_outputs/ - Demo script outputs")
    print("• checkpoints/ - Trained models")
    
    return results


if __name__ == "__main__":
    main()