#!/usr/bin/env python3
"""
Test script for the Video Frame Iterator

This script tests the video frame iterator functionality without requiring
external dependencies beyond OpenCV and NumPy.
"""

import sys
import numpy as np
from pathlib import Path

# Test if we can import the video iterator
try:
    from video_processing import VideoFrameIterator, VideoFrameMetadata, BatchVideoProcessor, TensorFrameBatcher
    print("✓ Video iterator imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

def create_test_video(output_path: str = "test_video.mp4", duration_seconds: int = 5, fps: int = 30):
    """
    Create a simple test video for demonstration purposes
    """
    try:
        import cv2
        print("✓ OpenCV available")
        
        # Video properties
        width, height = 640, 480
        total_frames = duration_seconds * fps
        
        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"Creating test video: {output_path}")
        print(f"  Duration: {duration_seconds}s, FPS: {fps}, Total frames: {total_frames}")
        
        for frame_num in range(total_frames):
            # Create a simple animated frame
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Add moving circle
            center_x = int(width * (0.5 + 0.3 * np.sin(frame_num * 0.1)))
            center_y = int(height * (0.5 + 0.3 * np.cos(frame_num * 0.1)))
            
            # Color changes over time
            color = (
                int(128 + 127 * np.sin(frame_num * 0.05)),
                int(128 + 127 * np.sin(frame_num * 0.03 + 2)),
                int(128 + 127 * np.sin(frame_num * 0.07 + 4))
            )
            
            cv2.circle(frame, (center_x, center_y), 50, color, -1)
            
            # Add frame number text
            cv2.putText(frame, f"Frame {frame_num}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            out.write(frame)
        
        out.release()
        print(f"✓ Test video created: {output_path}")
        return output_path
        
    except ImportError:
        print("✗ OpenCV not available - cannot create test video")
        return None


def test_basic_iteration(video_path: str):
    """Test basic frame iteration"""
    print(f"\n=== Testing Basic Iteration ===")
    
    try:
        with VideoFrameIterator(video_path, frame_skip=10) as iterator:
            print(f"Video info: {iterator.video_info}")
            
            frame_count = 0
            for frame, metadata in iterator:
                print(f"Frame {metadata.frame_number}: {metadata.timestamp:.2f}s, "
                      f"Shape: {frame.shape}")
                
                # Test metadata access
                if metadata.custom_metadata:
                    brightness = metadata.custom_metadata.get('mean_brightness', 'N/A')
                    print(f"  Brightness: {brightness}")
                
                frame_count += 1
                if frame_count >= 5:  # Only process 5 frames for demo
                    break
            
            print(f"✓ Successfully processed {frame_count} frames")
            
    except Exception as e:
        print(f"✗ Error during basic iteration: {e}")


def test_specific_frame_access(video_path: str):
    """Test accessing specific frames"""
    print(f"\n=== Testing Specific Frame Access ===")
    
    try:
        with VideoFrameIterator(video_path) as iterator:
            # Test accessing frame at specific time
            frame, metadata = iterator.get_frame_at_time(1.5)  # 1.5 seconds
            print(f"✓ Frame at 1.5s: Frame #{metadata.frame_number}, Shape: {frame.shape}")
            
            # Test accessing frame by number
            frame, metadata = iterator.get_frame_at_number(30)
            print(f"✓ Frame #30: Timestamp {metadata.timestamp:.2f}s, Shape: {frame.shape}")
            
    except Exception as e:
        print(f"✗ Error during specific frame access: {e}")


def test_metadata_collection(video_path: str):
    """Test metadata collection and saving"""
    print(f"\n=== Testing Metadata Collection ===")
    
    try:
        metadata_list = []
        
        with VideoFrameIterator(
            video_path, 
            frame_skip=15, 
            collect_frame_stats=True,
            resize_frames=(320, 240)
        ) as iterator:
            
            for i, (frame, metadata) in enumerate(iterator):
                metadata_list.append(metadata)
                
                if i >= 10:  # Collect 10 frames worth of metadata
                    break
            
            # Save metadata
            iterator.save_metadata_batch(metadata_list, "test_metadata.json")
            print(f"✓ Collected and saved metadata for {len(metadata_list)} frames")
            
            # Print sample metadata
            if metadata_list:
                sample = metadata_list[0].to_dict()
                print(f"Sample metadata keys: {list(sample.keys())}")
                if 'custom_metadata' in sample:
                    print(f"Custom metadata keys: {list(sample['custom_metadata'].keys())}")
            
    except Exception as e:
        print(f"✗ Error during metadata collection: {e}")


def test_batch_processing(video_paths: list):
    """Test batch processing of multiple videos"""
    print(f"\n=== Testing Batch Processing ===")
    
    try:
        processor = BatchVideoProcessor(
            video_paths,
            frame_skip=20,
            collect_frame_stats=True
        )
        
        results = processor.process_all()
        
        print(f"✓ Batch processing completed for {len(results)} videos")
        for video_path, metadata_list in results.items():
            print(f"  {Path(video_path).name}: {len(metadata_list)} frames")
            
    except Exception as e:
        print(f"✗ Error during batch processing: {e}")


def test_tensor_functionality(video_path: str):
    """Test tensor output functionality"""
    print(f"\n=== Testing Tensor Functionality ===")
    
    try:
        # Test tensor output
        with VideoFrameIterator(
            video_path, 
            frame_skip=20,
            output_format="tensor",
            normalize=True,
            resize_frames=(224, 224)
        ) as iterator:
            
            for i, (frame_tensor, metadata) in enumerate(iterator):
                print(f"Tensor frame {metadata.frame_number}: "
                      f"Shape {frame_tensor.shape}, "
                      f"dtype {frame_tensor.dtype}, "
                      f"range [{frame_tensor.min():.3f}, {frame_tensor.max():.3f}]")
                
                if i >= 3:  # Test 3 frames
                    break
        
        print("✓ Tensor output working")
        
        # Test tensor batching
        print("\n--- Testing Tensor Batching ---")
        batcher = TensorFrameBatcher(batch_size=3, device="cpu")
        
        with VideoFrameIterator(
            video_path,
            frame_skip=25,
            output_format="tensor",
            normalize=True,
            resize_frames=(224, 224)
        ) as iterator:
            
            for i, (frame_tensor, metadata) in enumerate(iterator):
                batch_tensor, batch_metadata = batcher.add_frame(frame_tensor, metadata)
                
                if batch_tensor is not None:
                    print(f"Batch created: Shape {batch_tensor.shape}")
                    break
                
                if i >= 5:  # Safety limit
                    break
        
        print("✓ Tensor batching working")
        
    except Exception as e:
        print(f"✗ Error during tensor testing: {e}")


def main():
    """Main test function"""
    print("=== Video Frame Iterator Test Suite ===")
    
    # Check if OpenCV is available
    try:
        import cv2
        print("✓ OpenCV available")
        opencv_available = True
    except ImportError:
        print("✗ OpenCV not available - limited testing possible")
        opencv_available = False
    
    # Check if PyTorch is available
    try:
        import torch
        print("✓ PyTorch available")
        torch_available = True
    except ImportError:
        print("✗ PyTorch not available - tensor functionality disabled")
        torch_available = False
    
    if opencv_available:
        # Create test video
        test_video_path = create_test_video()
        
        if test_video_path and Path(test_video_path).exists():
            # Run basic tests
            test_basic_iteration(test_video_path)
            test_specific_frame_access(test_video_path)
            test_metadata_collection(test_video_path)
            test_batch_processing([test_video_path])
            
            # Test tensor functionality if PyTorch is available
            if torch_available:
                test_tensor_functionality(test_video_path)
            else:
                print("\n⚠️  Skipping tensor tests - PyTorch not available")
            
            print(f"\n=== Test Summary ===")
            print(f"✓ All available tests completed")
            print(f"Test video: {test_video_path}")
            print(f"Metadata file: test_metadata.json")
            
            if torch_available:
                print("✓ Tensor functionality tested")
            else:
                print("⚠️  Tensor functionality not tested (install PyTorch)")
            
            # Clean up
            try:
                Path(test_video_path).unlink()
                print(f"✓ Cleaned up test video")
            except:
                print(f"Note: Test video {test_video_path} left for manual inspection")
        
        else:
            print("✗ Could not create test video")
    
    else:
        print("\nTo fully test this iterator, install dependencies:")
        print("  pip install opencv-python numpy")
        print("  pip install torch  # For tensor functionality")
        print("\nThe iterator is ready to use once dependencies are available.")


if __name__ == "__main__":
    main()