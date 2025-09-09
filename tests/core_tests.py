#!/usr/bin/env python3
"""
Core functionality tests for U-Net project

This module consolidates tests for:
- Environment setup and package imports
- Dataset functionality (BDD100K, KITTI)
- Video iterator and processing
- Basic video file operations
"""

import sys
import os
import torch
import numpy as np
import cv2
from pathlib import Path
import time
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from video_processing.frame_iterator import VideoFrameIterator, TensorFrameBatcher
from data.bdd100k_dataset import BDD100KSegmentationDataset
from data.kitti_dataset import KITTISegmentationDataset
from data.dataset_utils import create_sample_dataset_for_testing


class CoreTester:
    """Test core functionality of the U-Net project"""
    
    def __init__(self, output_dir: str = "test_outputs/core_tests"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
    
    def test_environment_setup(self) -> Dict[str, Any]:
        """Test environment setup and package imports"""
        print("🔧 Testing Environment Setup")
        print("=" * 40)
        
        results = {
            'imports_successful': True,
            'torch_available': False,
            'cv2_available': False,
            'numpy_available': False,
            'errors': []
        }
        
        # Test PyTorch
        try:
            import torch
            print(f"✅ PyTorch: {torch.__version__}")
            print(f"   CUDA available: {torch.cuda.is_available()}")
            results['torch_available'] = True
        except ImportError as e:
            print(f"❌ PyTorch import failed: {e}")
            results['errors'].append(f"PyTorch: {e}")
            results['imports_successful'] = False
        
        # Test OpenCV
        try:
            import cv2
            print(f"✅ OpenCV: {cv2.__version__}")
            results['cv2_available'] = True
        except ImportError as e:
            print(f"❌ OpenCV import failed: {e}")
            results['errors'].append(f"OpenCV: {e}")
            results['imports_successful'] = False
        
        # Test NumPy
        try:
            import numpy as np
            print(f"✅ NumPy: {np.__version__}")
            results['numpy_available'] = True
        except ImportError as e:
            print(f"❌ NumPy import failed: {e}")
            results['errors'].append(f"NumPy: {e}")
            results['imports_successful'] = False
        
        # Test project modules
        try:
            from models.unet import UNet
            print("✅ U-Net model import successful")
        except ImportError as e:
            print(f"❌ U-Net model import failed: {e}")
            results['errors'].append(f"U-Net: {e}")
            results['imports_successful'] = False
        
        print(f"\nEnvironment test {'PASSED' if results['imports_successful'] else 'FAILED'}")
        return results
    
    def test_dataset_functionality(self) -> Dict[str, Any]:
        """Test dataset creation and functionality"""
        print("\n📊 Testing Dataset Functionality")
        print("=" * 40)
        
        results = {
            'sample_dataset_created': False,
            'bdd100k_dataset_created': False,
            'kitti_dataset_created': False,
            'errors': []
        }
        
        # Test sample dataset creation
        try:
            print("Creating sample dataset...")
            sample_data = create_sample_dataset_for_testing(output_dir='data')
            print(f"✅ Sample dataset created: {sample_data}")
            results['sample_dataset_created'] = True
        except Exception as e:
            print(f"❌ Sample dataset creation failed: {e}")
            results['errors'].append(f"Sample dataset: {e}")
        
        # Test BDD100K dataset
        try:
            print("Testing BDD100K dataset...")
            dataset = BDD100KSegmentationDataset(
                data_dir="data/sample_bdd100k",
                split="train",
                image_size=(256, 256),
                augmentation=True
            )
            print(f"✅ BDD100K dataset created with {len(dataset)} samples")
            results['bdd100k_dataset_created'] = True
            
            # Test data loading
            if len(dataset) > 0:
                image, mask = dataset[0]
                print(f"   Sample shape: {image.shape}, {mask.shape}")
                
        except Exception as e:
            print(f"❌ BDD100K dataset failed: {e}")
            results['errors'].append(f"BDD100K: {e}")
        
        # Test KITTI dataset
        try:
            print("Testing KITTI dataset...")
            dataset = KITTISegmentationDataset(
                data_dir="data/sample_kitti",
                split="train",
                image_size=(256, 256)
            )
            print(f"✅ KITTI dataset created with {len(dataset)} samples")
            results['kitti_dataset_created'] = True
        except Exception as e:
            print(f"❌ KITTI dataset failed: {e}")
            results['errors'].append(f"KITTI: {e}")
        
        return results
    
    def test_video_iterator(self, video_path: Optional[str] = None) -> Dict[str, Any]:
        """Test video iterator functionality"""
        print("\n🎬 Testing Video Iterator")
        print("=" * 40)
        
        results = {
            'iterator_created': False,
            'frames_processed': 0,
            'tensor_batching': False,
            'errors': []
        }
        
        # Create test video if none provided
        if video_path is None:
            video_path = self._create_test_video()
        
        if not os.path.exists(video_path):
            results['errors'].append(f"Video file not found: {video_path}")
            return results
        
        try:
            # Test basic iteration
            print(f"Testing video: {video_path}")
            iterator = VideoFrameIterator(video_path, frame_skip=10, max_frames=5)
            
            frame_count = 0
            for frame, metadata in iterator:
                frame_count += 1
                print(f"  Frame {metadata.frame_number}: {frame.shape}")
            
            print(f"✅ Processed {frame_count} frames")
            results['iterator_created'] = True
            results['frames_processed'] = frame_count
            
        except Exception as e:
            print(f"❌ Video iteration failed: {e}")
            results['errors'].append(f"Video iteration: {e}")
        
        # Test tensor batching
        try:
            print("Testing tensor batching...")
            batcher = TensorFrameBatcher(batch_size=2, target_size=(256, 256))
            
            iterator = VideoFrameIterator(video_path, frame_skip=15, max_frames=4)
            batch_count = 0
            
            for frame, metadata in iterator:
                batch_tensor = batcher.add_frame(frame, metadata)
                if batch_tensor is not None:
                    batch_count += 1
                    print(f"  Batch {batch_count}: {batch_tensor.shape}")
            
            print(f"✅ Created {batch_count} batches")
            results['tensor_batching'] = True
            
        except Exception as e:
            print(f"❌ Tensor batching failed: {e}")
            results['errors'].append(f"Tensor batching: {e}")
        
        return results
    
    def test_video_file_operations(self, video_path: Optional[str] = None) -> Dict[str, Any]:
        """Test video file operations"""
        print("\n📁 Testing Video File Operations")
        print("=" * 40)
        
        results = {
            'video_readable': False,
            'video_info_extracted': False,
            'frame_extraction': False,
            'errors': []
        }
        
        if video_path is None:
            video_path = self._create_test_video()
        
        if not os.path.exists(video_path):
            results['errors'].append(f"Video file not found: {video_path}")
            return results
        
        try:
            # Test video reading
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                print(f"✅ Video opened successfully: {video_path}")
                results['video_readable'] = True
                
                # Extract video info
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                print(f"   FPS: {fps}")
                print(f"   Frame count: {frame_count}")
                print(f"   Resolution: {width}x{height}")
                results['video_info_extracted'] = True
                
                # Test frame extraction
                ret, frame = cap.read()
                if ret:
                    print(f"✅ Frame extracted: {frame.shape}")
                    results['frame_extraction'] = True
                    
                    # Save test frame
                    test_frame_path = self.output_dir / "test_frame.jpg"
                    cv2.imwrite(str(test_frame_path), frame)
                    print(f"   Test frame saved: {test_frame_path}")
                
                cap.release()
            else:
                results['errors'].append("Failed to open video")
                
        except Exception as e:
            print(f"❌ Video file operations failed: {e}")
            results['errors'].append(f"Video operations: {e}")
        
        return results
    
    def _create_test_video(self, output_path: str = "test_video.mp4", duration_seconds: int = 5, fps: int = 30) -> str:
        """Create a test video for testing purposes"""
        print(f"Creating test video: {output_path}")
        
        # Video properties
        width, height = 640, 480
        total_frames = duration_seconds * fps
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Generate frames
        for frame_num in range(total_frames):
            # Create a simple animated frame
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Add moving rectangle
            x = int((frame_num / total_frames) * (width - 100))
            y = height // 2 - 50
            cv2.rectangle(frame, (x, y), (x + 100, y + 100), (0, 255, 0), -1)
            
            # Add frame number text
            cv2.putText(frame, f"Frame {frame_num}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            out.write(frame)
        
        out.release()
        print(f"✅ Test video created: {output_path}")
        return output_path
    
    def run_all_tests(self, video_path: Optional[str] = None) -> Dict[str, Any]:
        """Run all core tests"""
        print("🧪 Running Core Functionality Tests")
        print("=" * 50)
        
        # Run individual tests
        env_results = self.test_environment_setup()
        dataset_results = self.test_dataset_functionality()
        video_results = self.test_video_iterator(video_path)
        file_results = self.test_video_file_operations(video_path)
        
        # Compile results
        all_results = {
            'environment': env_results,
            'datasets': dataset_results,
            'video_iterator': video_results,
            'video_files': file_results,
            'overall_success': True
        }
        
        # Check overall success
        for test_name, test_results in all_results.items():
            if test_name == 'overall_success':
                continue
            if test_results.get('errors') and len(test_results['errors']) > 0:
                all_results['overall_success'] = False
                break
        
        # Print summary
        print(f"\n📋 Test Summary")
        print("=" * 50)
        print(f"Environment: {'✅' if env_results['imports_successful'] else '❌'}")
        print(f"Datasets: {'✅' if dataset_results['sample_dataset_created'] else '❌'}")
        print(f"Video Iterator: {'✅' if video_results['iterator_created'] else '❌'}")
        print(f"Video Files: {'✅' if file_results['video_readable'] else '❌'}")
        print(f"\nOverall: {'✅ PASSED' if all_results['overall_success'] else '❌ FAILED'}")
        
        return all_results


def main():
    """Run core tests"""
    tester = CoreTester()
    
    # Check if video file exists
    video_path = "video_processing/data3_users_yoavnavon_clips_dqa_glued_sv_vehicle_day_FRONT_RIGHT_GENERIC_1x.mp4"
    if not os.path.exists(video_path):
        video_path = None  # Will create test video
    
    results = tester.run_all_tests(video_path)
    
    # Save results
    results_file = tester.output_dir / "core_test_results.txt"
    with open(results_file, 'w') as f:
        f.write("Core Functionality Test Results\n")
        f.write("=" * 40 + "\n\n")
        
        for test_name, test_results in results.items():
            if test_name == 'overall_success':
                continue
            f.write(f"{test_name.upper()}:\n")
            for key, value in test_results.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
        
        f.write(f"Overall Success: {results['overall_success']}\n")
    
    print(f"\n📄 Results saved to: {results_file}")


if __name__ == "__main__":
    main()
