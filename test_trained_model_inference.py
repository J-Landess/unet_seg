#!/usr/bin/env python3
"""
Test the trained U-Net model on unseen video data
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

class SimpleUNet(nn.Module):
    """Simple U-Net implementation (same as training)"""
    
    def __init__(self, in_channels=3, num_classes=21):
        super(SimpleUNet, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        
        # Final classification
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(torch.max_pool2d(enc1, 2))
        enc3 = self.enc3(torch.max_pool2d(enc2, 2))
        enc4 = self.enc4(torch.max_pool2d(enc3, 2))
        
        # Bottleneck
        bottleneck = self.bottleneck(torch.max_pool2d(enc4, 2))
        
        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)
        
        return self.final(dec1)

def load_trained_model(model_path="checkpoints/best_model.pth", num_classes=21):
    """Load the trained model"""
    print(f"🔄 Loading trained model from {model_path}...")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Create model
    model = SimpleUNet(in_channels=3, num_classes=num_classes)
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    print(f"✅ Model loaded successfully!")
    return model

def preprocess_frame(frame, target_size=(256, 256)):
    """Preprocess frame for inference"""
    # Resize frame
    frame_resized = cv2.resize(frame, target_size)
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    frame_normalized = frame_rgb.astype(np.float32) / 255.0
    
    # Convert to tensor and add batch dimension
    frame_tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1).unsqueeze(0)
    
    return frame_tensor, frame_resized

def postprocess_prediction(prediction, original_size):
    """Postprocess prediction back to original size"""
    # Get class predictions
    pred_classes = torch.argmax(prediction, dim=1).squeeze(0).numpy()
    
    # Resize back to original size
    pred_resized = cv2.resize(pred_classes.astype(np.uint8), original_size, interpolation=cv2.INTER_NEAREST)
    
    return pred_resized

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

def test_model_on_video(
    video_path,
    model_path="checkpoints/best_model.pth",
    output_dir="trained_model_inference_output",
    frame_skip=30,
    max_frames=10
):
    """Test the trained model on video data"""
    
    print("🎬 Testing Trained U-Net Model on Video")
    print("=" * 50)
    print(f"Video: {video_path}")
    print(f"Model: {model_path}")
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
    
    # Load trained model
    model = load_trained_model(model_path)
    
    # Define class colors for visualization
    class_colors = [
        (0, 0, 0),      # Background - Black
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
        resize_frames=None  # Keep original resolution
    )
    
    # Process frames
    print("🔄 Running inference on video frames...")
    frame_count = 0
    start_time = time.time()
    
    with torch.no_grad():
        for frame, metadata in iterator:
            frame_count += 1
            print(f"Processing frame {metadata.frame_number} ({frame_count}/{max_frames})...")
            
            # Preprocess frame
            frame_tensor, frame_resized = preprocess_frame(frame, target_size=(256, 256))
            
            # Run inference
            prediction = model(frame_tensor)
            
            # Postprocess prediction
            pred_mask = postprocess_prediction(prediction, (frame.shape[1], frame.shape[0]))
            
            # Create visualization
            vis_frame = create_visualization(frame, pred_mask, class_colors)
            
            # Save results
            base_name = f"frame_{metadata.frame_number:06d}"
            
            # Save original frame
            orig_path = os.path.join(output_dir, f"{base_name}_original.jpg")
            cv2.imwrite(orig_path, frame)
            
            # Save prediction mask
            mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
            cv2.imwrite(mask_path, pred_mask.astype(np.uint8) * 10)  # Scale for visibility
            
            # Save visualization
            vis_path = os.path.join(output_dir, f"{base_name}_visualization.jpg")
            cv2.imwrite(vis_path, vis_frame)
            
            # Save overlay (original + mask)
            overlay = cv2.addWeighted(frame, 0.7, vis_frame, 0.3, 0)
            overlay_path = os.path.join(output_dir, f"{base_name}_overlay.jpg")
            cv2.imwrite(overlay_path, overlay)
            
            # Save resized frame with prediction
            resized_vis = create_visualization(frame_resized, pred_mask, class_colors)
            resized_overlay = cv2.addWeighted(frame_resized, 0.7, resized_vis, 0.3, 0)
            resized_path = os.path.join(output_dir, f"{base_name}_resized_overlay.jpg")
            cv2.imwrite(resized_path, resized_overlay)
            
            if frame_count >= max_frames:
                break
    
    # Calculate timing
    end_time = time.time()
    total_time = end_time - start_time
    fps = frame_count / total_time if total_time > 0 else 0
    
    print(f"\n✅ Inference completed!")
    print(f"   Processed {frame_count} frames in {total_time:.2f} seconds")
    print(f"   Average FPS: {fps:.2f}")
    print(f"   Output saved to: {output_dir}")
    
    # Create summary
    create_summary(output_dir, frame_count, model_path)

def create_summary(output_dir, frame_count, model_path):
    """Create a summary file with results"""
    summary_path = os.path.join(output_dir, "inference_summary.txt")
    
    with open(summary_path, 'w') as f:
        f.write("Trained Model Inference Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Model used: {model_path}\n")
        f.write(f"Frames processed: {frame_count}\n")
        f.write(f"Model type: Trained U-Net (21 classes)\n\n")
        f.write("Output files per frame:\n")
        f.write("- *_original.jpg: Original video frame\n")
        f.write("- *_mask.png: Segmentation mask (scaled)\n")
        f.write("- *_visualization.jpg: Colored segmentation\n")
        f.write("- *_overlay.jpg: Original + segmentation overlay\n")
        f.write("- *_resized_overlay.jpg: Resized frame + segmentation\n\n")
        f.write("This shows how the trained model performs on unseen video data.\n")
    
    print(f"📄 Summary saved to: {summary_path}")

def main():
    """Main function"""
    video_path = "video_processing/data3_users_yoavnavon_clips_dqa_glued_sv_vehicle_day_FRONT_RIGHT_GENERIC_1x.mp4"
    model_path = "checkpoints/best_model.pth"
    
    test_model_on_video(
        video_path=video_path,
        model_path=model_path,
        output_dir="trained_model_inference_output",
        frame_skip=30,  # Process every 30th frame
        max_frames=8    # Process 8 frames total
    )

if __name__ == "__main__":
    main()
