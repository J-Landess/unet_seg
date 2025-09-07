#!/usr/bin/env python3
"""
Test script for video inference with the U-Net model
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
from inference.inference import SegmentationInference

def test_video_inference(
    video_path: str,
    model_path: str = None,
    config_path: str = None,
    output_dir: str = "video_inference_output",
    frame_skip: int = 10,
    max_frames: int = 50
):
    """
    Test video inference with the U-Net model
    
    Args:
        video_path: Path to input video
        model_path: Path to trained model (optional, will use dummy if not provided)
        config_path: Path to config file (optional, will use default if not provided)
        output_dir: Directory to save output frames
        frame_skip: Process every Nth frame
        max_frames: Maximum number of frames to process
    """
    
    print("Video Inference Test")
    print("=" * 50)
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
    
    # Check if we have a real model or need to use dummy inference
    use_dummy = model_path is None or not Path(model_path).exists()
    
    if use_dummy:
        print("⚠️  No trained model found, using dummy inference for demonstration")
        print("   To use real inference, provide a trained model with --model path/to/model.pth")
        print()
        
        # Dummy inference function
        def dummy_predict(frame):
            # Create a dummy segmentation mask
            h, w = frame.shape[:2]
            # Create a simple pattern based on image intensity
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mask = np.zeros((h, w), dtype=np.uint8)
            
            # Simple threshold-based segmentation
            mask[gray > 128] = 1  # Bright areas
            mask[gray > 200] = 2  # Very bright areas
            
            # Add some geometric patterns
            center_x, center_y = w // 2, h // 2
            y, x = np.ogrid[:h, :w]
            mask[(x - center_x)**2 + (y - center_y)**2 < (min(w, h) // 4)**2] = 3
            
            return mask, np.random.random((3, h, w))  # Dummy probabilities
        
        inference_func = dummy_predict
        class_names = ["Background", "Object1", "Object2", "Circle"]
        class_colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)]
        
    else:
        print(f"✅ Using trained model: {model_path}")
        print(f"✅ Using config: {config_path}")
        print()
        
        # Initialize real inference
        inference = SegmentationInference(model_path, config_path)
        
        def real_predict(frame):
            prediction, probabilities = inference.predict_single(frame)
            return prediction, probabilities
        
        inference_func = real_predict
        class_names = getattr(inference, 'class_names', [f"Class_{i}" for i in range(10)])
        class_colors = getattr(inference, 'class_colors', [(i*50, i*30, i*20) for i in range(10)])
    
    # Initialize video iterator
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
    
    for frame, metadata in iterator:
        frame_count += 1
        print(f"Processing frame {metadata.frame_number} ({frame_count}/{max_frames})...")
        
        # Run inference
        prediction, probabilities = inference_func(frame)
        
        # Create visualization
        vis_frame = create_visualization(frame, prediction, class_colors)
        
        # Save results
        base_name = f"frame_{metadata.frame_number:06d}"
        
        # Save original frame
        orig_path = os.path.join(output_dir, f"{base_name}_original.jpg")
        cv2.imwrite(orig_path, frame)
        
        # Save prediction mask
        mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
        cv2.imwrite(mask_path, prediction.astype(np.uint8) * 50)  # Scale for visibility
        
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
    create_summary(output_dir, frame_count, class_names, use_dummy)

def create_visualization(frame, prediction, class_colors):
    """Create colored visualization of the prediction"""
    h, w = frame.shape[:2]
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, color in enumerate(class_colors):
        mask = prediction == class_id
        vis[mask] = color
    
    return vis

def create_summary(output_dir, frame_count, class_names, use_dummy):
    """Create a summary file with results"""
    summary_path = os.path.join(output_dir, "inference_summary.txt")
    
    with open(summary_path, 'w') as f:
        f.write("Video Inference Summary\n")
        f.write("=" * 30 + "\n\n")
        f.write(f"Frames processed: {frame_count}\n")
        f.write(f"Model type: {'Dummy (demo)' if use_dummy else 'Trained model'}\n")
        f.write(f"Classes: {', '.join(class_names)}\n\n")
        f.write("Output files per frame:\n")
        f.write("- *_original.jpg: Original video frame\n")
        f.write("- *_mask.png: Segmentation mask\n")
        f.write("- *_visualization.jpg: Colored segmentation\n")
        f.write("- *_overlay.jpg: Original + segmentation overlay\n")
    
    print(f"📄 Summary saved to: {summary_path}")

def main():
    """Main function with command line arguments"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test video inference with U-Net")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--model", help="Path to trained model (.pth file)")
    parser.add_argument("--config", help="Path to config file (.yaml)")
    parser.add_argument("--output", default="video_inference_output", help="Output directory")
    parser.add_argument("--frame-skip", type=int, default=10, help="Process every Nth frame")
    parser.add_argument("--max-frames", type=int, default=50, help="Maximum frames to process")
    
    args = parser.parse_args()
    
    test_video_inference(
        video_path=args.video,
        model_path=args.model,
        config_path=args.config,
        output_dir=args.output,
        frame_skip=args.frame_skip,
        max_frames=args.max_frames
    )

if __name__ == "__main__":
    main()
