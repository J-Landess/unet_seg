#!/usr/bin/env python3
"""
Simple video inference test that works with the current codebase structure
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from video_processing.frame_iterator import VideoFrameIterator

def test_video_inference_simple(
    video_path: str,
    output_dir: str = "video_inference_output",
    frame_skip: int = 15,
    max_frames: int = 10
):
    """
    Simple video inference test with dummy segmentation
    """
    
    print("Simple Video Inference Test")
    print("=" * 40)
    print(f"Video: {video_path}")
    print(f"Output: {output_dir}")
    print(f"Frame skip: {frame_skip}")
    print(f"Max frames: {max_frames}")
    print()
    
    # Check if video exists
    if not Path(video_path).exists():
        print(f"❌ Error: Video file not found: {video_path}")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("🎬 Initializing video iterator...")
    iterator = VideoFrameIterator(
        video_path=video_path,
        frame_skip=frame_skip,
        start_frame=0,
        end_frame=max_frames * frame_skip,
        collect_frame_stats=True,
        output_format="numpy",
        resize_frames=None  # Keep original resolution
    )
    
    # Process frames
    print("🔄 Processing video frames...")
    frame_count = 0
    start_time = time.time()
    
    # Define class colors for visualization
    class_colors = [
        (0, 0, 0),      # Background - Black
        (255, 0, 0),    # Class 1 - Red
        (0, 255, 0),    # Class 2 - Green
        (0, 0, 255),    # Class 3 - Blue
        (255, 255, 0),  # Class 4 - Yellow
        (255, 0, 255),  # Class 5 - Magenta
        (0, 255, 255),  # Class 6 - Cyan
    ]
    
    for frame, metadata in iterator:
        frame_count += 1
        print(f"Processing frame {metadata.frame_number} ({frame_count}/{max_frames})...")
        
        # Create dummy segmentation based on image features
        prediction = create_dummy_segmentation(frame)
        
        # Create visualization
        vis_frame = create_visualization(frame, prediction, class_colors)
        
        # Save results
        base_name = f"frame_{metadata.frame_number:06d}"
        
        # Save original frame
        orig_path = os.path.join(output_dir, f"{base_name}_original.jpg")
        cv2.imwrite(orig_path, frame)
        
        # Save prediction mask (scaled for visibility)
        mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
        cv2.imwrite(mask_path, prediction.astype(np.uint8) * 40)  # Scale for visibility
        
        # Save visualization
        vis_path = os.path.join(output_dir, f"{base_name}_visualization.jpg")
        cv2.imwrite(vis_path, vis_frame)
        
        # Save overlay (original + mask)
        overlay = cv2.addWeighted(frame, 0.7, vis_frame, 0.3, 0)
        overlay_path = os.path.join(output_dir, f"{base_name}_overlay.jpg")
        cv2.imwrite(overlay_path, overlay)
        
        if frame_count >= max_frames:
            break
    
    # Calculate timing
    end_time = time.time()
    total_time = end_time - start_time
    fps = frame_count / total_time if total_time > 0 else 0
    
    print(f"\n✅ Video inference completed!")
    print(f"   Processed {frame_count} frames in {total_time:.2f} seconds")
    print(f"   Average FPS: {fps:.2f}")
    print(f"   Output saved to: {output_dir}")
    
    # Create summary
    create_summary(output_dir, frame_count)

def create_dummy_segmentation(frame):
    """Create a dummy segmentation mask based on image features"""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Initialize mask
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Class 1: Dark areas
    mask[gray < 50] = 1
    
    # Class 2: Medium dark areas
    mask[(gray >= 50) & (gray < 100)] = 2
    
    # Class 3: Medium areas
    mask[(gray >= 100) & (gray < 150)] = 3
    
    # Class 4: Bright areas
    mask[(gray >= 150) & (gray < 200)] = 4
    
    # Class 5: Very bright areas
    mask[gray >= 200] = 5
    
    # Add some geometric patterns
    center_x, center_y = w // 2, h // 2
    
    # Circle in center
    y, x = np.ogrid[:h, :w]
    circle_mask = (x - center_x)**2 + (y - center_y)**2 < (min(w, h) // 6)**2
    mask[circle_mask] = 6
    
    # Horizontal and vertical lines
    mask[h//4, :] = 1  # Horizontal line
    mask[:, w//4] = 2  # Vertical line
    
    return mask

def create_visualization(frame, prediction, class_colors):
    """Create colored visualization of the prediction"""
    h, w = frame.shape[:2]
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, color in enumerate(class_colors):
        if class_id < len(class_colors):
            mask = prediction == class_id
            vis[mask] = color
    
    return vis

def create_summary(output_dir, frame_count):
    """Create a summary file with results"""
    summary_path = os.path.join(output_dir, "inference_summary.txt")
    
    with open(summary_path, 'w') as f:
        f.write("Video Inference Summary\n")
        f.write("=" * 30 + "\n\n")
        f.write(f"Frames processed: {frame_count}\n")
        f.write("Model type: Dummy segmentation (demo)\n")
        f.write("Classes: Background, Dark, Medium-Dark, Medium, Bright, Very-Bright, Circle\n\n")
        f.write("Output files per frame:\n")
        f.write("- *_original.jpg: Original video frame\n")
        f.write("- *_mask.png: Segmentation mask (scaled)\n")
        f.write("- *_visualization.jpg: Colored segmentation\n")
        f.write("- *_overlay.jpg: Original + segmentation overlay\n\n")
        f.write("Note: This is a demonstration with dummy segmentation.\n")
        f.write("To use real U-Net inference, you need a trained model.\n")
    
    print(f"📄 Summary saved to: {summary_path}")

def main():
    # Use new test package
    from tests.test_video_inference import VideoInferenceTester
    
    tester = VideoInferenceTester()
    video_path = "video_processing/data3_users_yoavnavon_clips_dqa_glued_sv_vehicle_day_FRONT_RIGHT_GENERIC_1x.mp4"
    
    # Run tests using new package
    tester.test_dummy_inference(video_path, max_frames=10)
    return
    
    # Original code below (commented out)
    # 
    """Main function"""
    video_path = "video_processing/data3_users_yoavnavon_clips_dqa_glued_sv_vehicle_day_FRONT_RIGHT_GENERIC_1x.mp4"
    
    test_video_inference_simple(
        video_path=video_path,
        output_dir="video_inference_output",
        frame_skip=15,
        max_frames=10
    )

if __name__ == "__main__":
    main()
