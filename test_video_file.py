#!/usr/bin/env python3
"""
Test script for the video frame iterator with a specific video file
"""

import sys
import cv2
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from video_processing.frame_iterator import VideoFrameIterator, VideoFrameMetadata, TensorFrameBatcher

def test_video_iterator(video_path: str):
    """Test the video iterator with the specified video file"""
    
    print(f"Testing video iterator with: {video_path}")
    print("=" * 60)
    
    # Check if video file exists
    if not Path(video_path).exists():
        print(f"Error: Video file not found: {video_path}")
        return
    
    try:
        # Test 1: Basic frame iteration (numpy format)
        print("\n1. Testing basic frame iteration (numpy format)...")
        iterator = VideoFrameIterator(
            video_path=video_path,
            frame_skip=5,  # Process every 5th frame
            start_frame=0,
            end_frame=50,  # Limit to first 50 frames for testing
            collect_frame_stats=True,
            output_format="numpy"
        )
        
        frame_count = 0
        for frame, metadata in iterator:
            frame_count += 1
            print(f"Frame {metadata.frame_number}: shape={frame.shape}, timestamp={metadata.timestamp:.2f}s")
            
            if frame_count >= 5:  # Show first 5 frames
                break
        
        print(f"Successfully processed {frame_count} frames")
        
        # Test 2: Tensor format with batching
        print("\n2. Testing tensor format with batching...")
        iterator_tensor = VideoFrameIterator(
            video_path=video_path,
            frame_skip=10,  # Process every 10th frame
            start_frame=0,
            end_frame=100,  # Limit to first 100 frames
            collect_frame_stats=True,
            output_format="tensor",
            normalize=True,
            device="cpu"
        )
        
        batcher = TensorFrameBatcher(batch_size=4)
        batch_count = 0
        
        for frame_tensor, metadata in iterator_tensor:
            batch_tensor, batch_metadata = batcher.add_frame(frame_tensor, metadata)
            
            if batch_tensor is not None:  # Batch is ready
                batch_count += 1
                print(f"Batch {batch_count}: shape={batch_tensor.shape}, frames={len(batch_metadata)}")
                
                if batch_count >= 3:  # Show first 3 batches
                    break
        
        # Test 3: Video information
        print("\n3. Video information...")
        cap = iterator.cap
        if cap:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count_total / fps if fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"Video properties:")
            print(f"  - Resolution: {width}x{height}")
            print(f"  - FPS: {fps:.2f}")
            print(f"  - Total frames: {frame_count_total}")
            print(f"  - Duration: {duration:.2f} seconds")
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function"""
    video_path = "video_processing/data3_users_yoavnavon_clips_dqa_glued_sv_vehicle_day_FRONT_RIGHT_GENERIC_1x.mp4"
    
    print("Video Frame Iterator Test")
    print("=" * 40)
    
    test_video_iterator(video_path)

if __name__ == "__main__":
    main()
