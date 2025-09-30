#!/usr/bin/env python3
"""
Test pre-trained U-Net models with your video data
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from video_processing.frame_iterator import VideoFrameIterator

def install_pretrained_dependencies():
    """Install required packages for pre-trained models"""
    import subprocess
    import sys
    
    packages = [
        "segmentation-models-pytorch",
        "timm",
        "efficientnet-pytorch"
    ]
    
    for package in packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package} already installed")
        except ImportError:
            print(f"📦 Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def create_pretrained_unet(num_classes=21, encoder_name="vgg11", pretrained=True):
    """Create a pre-trained U-Net model using segmentation-models-pytorch"""
    try:
        from segmentation_models_pytorch import Unet
        
        model = Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet" if pretrained else None,
            classes=num_classes,
            activation=None,
            in_channels=3
        )
        
        print(f"✅ Created pre-trained U-Net with {encoder_name} encoder")
        return model
        
    except ImportError:
        print("❌ segmentation-models-pytorch not installed")
        print("Run: pip install segmentation-models-pytorch")
        return None

def create_efficientnet_unet(num_classes=21, pretrained=True):
    """Create U-Net with EfficientNet encoder"""
    try:
        from segmentation_models_pytorch import Unet
        
        model = Unet(
            encoder_name="efficientnet-b0",
            encoder_weights="imagenet" if pretrained else None,
            classes=num_classes,
            activation=None,
            in_channels=3
        )
        
        print(f"✅ Created EfficientNet U-Net")
        return model
        
    except ImportError:
        print("❌ segmentation-models-pytorch not installed")
        return None

def preprocess_frame_for_pretrained(frame, target_size=(512, 512)):
    """Preprocess frame for pre-trained models"""
    # Resize frame
    frame_resized = cv2.resize(frame, target_size)
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    frame_normalized = frame_rgb.astype(np.float32) / 255.0
    
    # Convert to tensor and add batch dimension
    frame_tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1).unsqueeze(0)
    
    return frame_tensor, frame_resized

def test_pretrained_model_on_video(
    video_path,
    model_type="vgg11",
    output_dir="pretrained_inference_output",
    frame_skip=30,
    max_frames=5
):
    """Test pre-trained model on video data"""
    
    print("🎬 Testing Pre-trained U-Net Model on Video")
    print("=" * 50)
    print(f"Video: {video_path}")
    print(f"Model: {model_type}")
    print(f"Output: {output_dir}")
    print()
    
    # Check if video exists
    if not Path(video_path).exists():
        print(f"❌ Error: Video file not found: {video_path}")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create pre-trained model
    if model_type == "vgg11":
        model = create_pretrained_unet(num_classes=21, encoder_name="vgg11", pretrained=True)
    elif model_type == "efficientnet":
        model = create_efficientnet_unet(num_classes=21, pretrained=True)
    else:
        print(f"❌ Unknown model type: {model_type}")
        return
    
    if model is None:
        print("❌ Failed to create pre-trained model")
        return
    
    # Move model to CPU and set to eval mode
    model = model.cpu()
    model.eval()
    
    print(f"🧠 Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Define class colors
    class_colors = [
        (0, 0, 0),      # Background
        (255, 0, 0),    # Class 1 - Red
        (0, 255, 0),    # Class 2 - Green
        (0, 0, 255),    # Class 3 - Blue
        (255, 255, 0),  # Class 4 - Yellow
        (255, 0, 255),  # Class 5 - Magenta
        (0, 255, 255),  # Class 6 - Cyan
        (128, 0, 128),  # Class 7 - Purple
        (255, 165, 0),  # Class 8 - Orange
        (0, 128, 0),    # Class 9 - Dark Green
        (128, 128, 128), # Class 10 - Gray
        (255, 192, 203), # Class 11 - Pink
        (165, 42, 42),  # Class 12 - Brown
        (0, 0, 128),    # Class 13 - Navy
        (128, 128, 0),  # Class 14 - Olive
        (0, 128, 128),  # Class 15 - Teal
        (128, 0, 0),    # Class 16 - Maroon
        (192, 192, 192), # Class 17 - Silver
        (255, 255, 255), # Class 18 - White
        (0, 0, 0),      # Class 19 - Black
        (64, 64, 64),   # Class 20 - Dark Gray
    ]
    
    # Initialize video iterator
    print("🎬 Initializing video iterator...")
    iterator = VideoFrameIterator(
        video_path=video_path,
        frame_skip=frame_skip,
        start_frame=0,
        end_frame=max_frames * frame_skip,
        collect_frame_stats=True,
        output_format="numpy",
        resize_frames=None
    )
    
    # Process frames
    print("🔄 Running inference with pre-trained model...")
    frame_count = 0
    start_time = time.time()
    
    with torch.no_grad():
        for frame, metadata in iterator:
            frame_count += 1
            print(f"Processing frame {metadata.frame_number} ({frame_count}/{max_frames})...")
            
            # Preprocess frame
            frame_tensor, frame_resized = preprocess_frame_for_pretrained(frame, target_size=(512, 512))
            
            # Run inference
            prediction = model(frame_tensor)
            
            # Get class predictions
            pred_classes = torch.argmax(prediction, dim=1).squeeze(0).numpy()
            
            # Resize prediction back to original size
            pred_resized = cv2.resize(pred_classes.astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            # Create visualization
            vis_frame = create_visualization(frame, pred_resized, class_colors)
            
            # Save results
            base_name = f"frame_{metadata.frame_number:06d}"
            
            # Save original frame
            orig_path = os.path.join(output_dir, f"{base_name}_original.jpg")
            cv2.imwrite(orig_path, frame)
            
            # Save prediction mask
            mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
            cv2.imwrite(mask_path, pred_resized.astype(np.uint8) * 10)
            
            # Save visualization
            vis_path = os.path.join(output_dir, f"{base_name}_visualization.jpg")
            cv2.imwrite(vis_path, vis_frame)
            
            # Save overlay
            overlay = cv2.addWeighted(frame, 0.7, vis_frame, 0.3, 0)
            overlay_path = os.path.join(output_dir, f"{base_name}_overlay.jpg")
            cv2.imwrite(overlay_path, overlay)
            
            if frame_count >= max_frames:
                break
    
    # Calculate timing
    end_time = time.time()
    total_time = end_time - start_time
    fps = frame_count / total_time if total_time > 0 else 0
    
    print(f"\n✅ Pre-trained model inference completed!")
    print(f"   Processed {frame_count} frames in {total_time:.2f} seconds")
    print(f"   Average FPS: {fps:.2f}")
    print(f"   Output saved to: {output_dir}")

def create_visualization(frame, prediction, class_colors):
    """Create colored visualization of the prediction"""
    h, w = frame.shape[:2]
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Ensure prediction matches frame dimensions
    if prediction.shape != (h, w):
        prediction = cv2.resize(prediction.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    
    for class_id, color in enumerate(class_colors):
        if class_id < len(class_colors):
            mask = prediction == class_id
            vis[mask] = color
    
    return vis

def main():
    """Main function"""
    print("🚀 Pre-trained U-Net Model Testing")
    print("=" * 40)
    
    # Install dependencies
    print("📦 Checking dependencies...")
    install_pretrained_dependencies()
    
    video_path = "video_processing/data3_users_yoavnavon_clips_dqa_glued_sv_vehicle_day_FRONT_RIGHT_GENERIC_1x.mp4"
    
    # Test with VGG11 encoder
    print("\n🔬 Testing with VGG11 encoder...")
    test_pretrained_model_on_video(
        video_path=video_path,
        model_type="vgg11",
        output_dir="pretrained_vgg11_output",
        frame_skip=30,
        max_frames=5
    )
    
    # Test with EfficientNet encoder
    print("\n🔬 Testing with EfficientNet encoder...")
    test_pretrained_model_on_video(
        video_path=video_path,
        model_type="efficientnet",
        output_dir="pretrained_efficientnet_output",
        frame_skip=30,
        max_frames=5
    )
    
    print("\n🎉 All tests completed!")
    print("Compare the results in:")
    print("- pretrained_vgg11_output/")
    print("- pretrained_efficientnet_output/")
    print("- trained_model_inference_output/ (your trained model)")

if __name__ == "__main__":
    main()
