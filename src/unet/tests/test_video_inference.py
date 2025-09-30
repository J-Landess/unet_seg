#!/usr/bin/env python3
"""
Video inference testing module
"""

import sys
import os
import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path
import time
from typing import Optional, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from video_processing.frame_iterator import VideoFrameIterator

class VideoInferenceTester:
    """Test video inference with different model types"""
    
    def __init__(self, output_base_dir: str = "test_outputs"):
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(exist_ok=True)
        
        # Class colors for visualization
        self.class_colors = [
            (0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 0, 128),
            (255, 165, 0), (0, 128, 0), (128, 128, 128), (255, 192, 203),
            (165, 42, 42), (0, 0, 128), (128, 128, 0), (0, 128, 128),
            (128, 0, 0), (192, 192, 192), (255, 255, 255), (0, 0, 0),
            (64, 64, 64)
        ]
    
    def test_dummy_inference(
        self, 
        video_path: str, 
        output_dir: str = "dummy_inference",
        frame_skip: int = 15,
        max_frames: int = 10
    ) -> Dict[str, Any]:
        """Test dummy segmentation on video"""
        
        print("🎭 Testing Dummy Segmentation")
        print("=" * 40)
        
        output_path = self.output_base_dir / output_dir
        output_path.mkdir(exist_ok=True)
        
        results = self._run_inference(
            video_path=video_path,
            model_type="dummy",
            output_dir=str(output_path),
            frame_skip=frame_skip,
            max_frames=max_frames
        )
        
        return results
    
    def test_trained_model_inference(
        self,
        video_path: str,
        model_path: str = "checkpoints/best_model.pth",
        output_dir: str = "trained_inference", 
        frame_skip: int = 30,
        max_frames: int = 8
    ) -> Dict[str, Any]:
        """Test trained model on video"""
        
        print("🧠 Testing Trained Model")
        print("=" * 40)
        
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            return {"error": "Model not found"}
        
        output_path = self.output_base_dir / output_dir
        output_path.mkdir(exist_ok=True)
        
        results = self._run_inference(
            video_path=video_path,
            model_type="trained",
            model_path=model_path,
            output_dir=str(output_path),
            frame_skip=frame_skip,
            max_frames=max_frames
        )
        
        return results
    
    def test_pretrained_inference(
        self,
        video_path: str,
        encoder_name: str = "vgg11",
        output_dir: str = "pretrained_inference",
        frame_skip: int = 30,
        max_frames: int = 5
    ) -> Dict[str, Any]:
        """Test pre-trained model on video"""
        
        print(f"🏗️ Testing Pre-trained Model ({encoder_name})")
        print("=" * 40)
        
        output_path = self.output_base_dir / output_dir
        output_path.mkdir(exist_ok=True)
        
        results = self._run_inference(
            video_path=video_path,
            model_type="pretrained",
            encoder_name=encoder_name,
            output_dir=str(output_path),
            frame_skip=frame_skip,
            max_frames=max_frames
        )
        
        return results
    
    def _run_inference(
        self,
        video_path: str,
        model_type: str,
        output_dir: str,
        frame_skip: int,
        max_frames: int,
        model_path: Optional[str] = None,
        encoder_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run inference with specified model type"""
        
        # Check if video exists
        if not Path(video_path).exists():
            return {"error": f"Video file not found: {video_path}"}
        
        # Initialize video iterator
        iterator = VideoFrameIterator(
            video_path=video_path,
            frame_skip=frame_skip,
            start_frame=0,
            end_frame=max_frames * frame_skip,
            collect_frame_stats=True,
            output_format="numpy",
            resize_frames=None
        )
        
        # Load model
        model = self._load_model(model_type, model_path, encoder_name)
        if model is None:
            return {"error": "Failed to load model"}
        
        # Process frames
        start_time = time.time()
        frame_count = 0
        
        with torch.no_grad():
            for frame, metadata in iterator:
                frame_count += 1
                print(f"Processing frame {metadata.frame_number} ({frame_count}/{max_frames})...")
                
                # Preprocess and predict
                prediction = self._predict_frame(model, frame, model_type)
                
                # Save results
                self._save_frame_results(
                    frame, prediction, metadata, output_dir, frame_count
                )
                
                if frame_count >= max_frames:
                    break
        
        # Calculate timing
        end_time = time.time()
        total_time = end_time - start_time
        fps = frame_count / total_time if total_time > 0 else 0
        
        results = {
            "model_type": model_type,
            "frames_processed": frame_count,
            "total_time": total_time,
            "fps": fps,
            "output_dir": output_dir
        }
        
        print(f"✅ {model_type.title()} inference completed!")
        print(f"   Processed {frame_count} frames in {total_time:.2f} seconds")
        print(f"   Average FPS: {fps:.2f}")
        
        return results
    
    def _load_model(self, model_type: str, model_path: Optional[str] = None, encoder_name: Optional[str] = None):
        """Load model based on type"""
        
        if model_type == "dummy":
            return "dummy"  # Special case for dummy model
        
        elif model_type == "trained":
            if model_path and os.path.exists(model_path):
                # Load your trained model
                from simple_training_test import SimpleUNet
                model = SimpleUNet(in_channels=3, num_classes=21)
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
                model.eval()
                return model
            return None
        
        elif model_type == "pretrained":
            try:
                from segmentation_models_pytorch import Unet
                
                # Fix encoder name for EfficientNet
                if encoder_name == "efficientnet":
                    encoder_name = "efficientnet-b0"
                
                model = Unet(
                    encoder_name=encoder_name,
                    encoder_weights="imagenet",
                    classes=21,
                    activation=None,
                    in_channels=3
                )
                model.eval()
                return model
            except ImportError:
                print("❌ segmentation-models-pytorch not installed")
                return None
        
        return None
    
    def _predict_frame(self, model, frame: np.ndarray, model_type: str) -> np.ndarray:
        """Predict segmentation for a single frame"""
        
        if model_type == "dummy":
            return self._create_dummy_prediction(frame)
        
        # Preprocess frame
        frame_tensor = self._preprocess_frame(frame)
        
        # Run inference
        prediction = model(frame_tensor)
        
        # Get class predictions
        pred_classes = torch.argmax(prediction, dim=1).squeeze(0).numpy()
        
        # Resize back to original size
        pred_resized = cv2.resize(
            pred_classes.astype(np.uint8), 
            (frame.shape[1], frame.shape[0]), 
            interpolation=cv2.INTER_NEAREST
        )
        
        return pred_resized
    
    def _create_dummy_prediction(self, frame: np.ndarray) -> np.ndarray:
        """Create dummy segmentation mask"""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[gray < 50] = 1
        mask[(gray >= 50) & (gray < 100)] = 2
        mask[(gray >= 100) & (gray < 150)] = 3
        mask[(gray >= 150) & (gray < 200)] = 4
        mask[gray >= 200] = 5
        
        # Add geometric patterns
        center_x, center_y = w // 2, h // 2
        y, x = np.ogrid[:h, :w]
        circle_mask = (x - center_x)**2 + (y - center_y)**2 < (min(w, h) // 6)**2
        mask[circle_mask] = 6
        
        return mask
    
    def _preprocess_frame(self, frame: np.ndarray, target_size: tuple = (512, 512)) -> torch.Tensor:
        """Preprocess frame for model inference"""
        # Resize frame
        frame_resized = cv2.resize(frame, target_size)
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        frame_normalized = frame_rgb.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        frame_tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1).unsqueeze(0)
        
        return frame_tensor
    
    def _save_frame_results(
        self, 
        frame: np.ndarray, 
        prediction: np.ndarray, 
        metadata, 
        output_dir: str, 
        frame_count: int
    ):
        """Save frame results"""
        base_name = f"frame_{metadata.frame_number:06d}"
        
        # Create visualization
        vis_frame = self._create_visualization(frame, prediction)
        
        # Save original frame
        orig_path = os.path.join(output_dir, f"{base_name}_original.jpg")
        cv2.imwrite(orig_path, frame)
        
        # Save prediction mask
        mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
        cv2.imwrite(mask_path, prediction.astype(np.uint8) * 10)
        
        # Save visualization
        vis_path = os.path.join(output_dir, f"{base_name}_visualization.jpg")
        cv2.imwrite(vis_path, vis_frame)
        
        # Save overlay
        overlay = cv2.addWeighted(frame, 0.7, vis_frame, 0.3, 0)
        overlay_path = os.path.join(output_dir, f"{base_name}_overlay.jpg")
        cv2.imwrite(overlay_path, overlay)
    
    def _create_visualization(self, frame: np.ndarray, prediction: np.ndarray) -> np.ndarray:
        """Create colored visualization of prediction"""
        h, w = frame.shape[:2]
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Ensure prediction matches frame dimensions
        if prediction.shape != (h, w):
            prediction = cv2.resize(prediction.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        
        for class_id, color in enumerate(self.class_colors):
            if class_id < len(self.class_colors):
                mask = prediction == class_id
                vis[mask] = color
        
        return vis

def main():
    """Example usage"""
    tester = VideoInferenceTester()
    
    video_path = "video_processing/data3_users_yoavnavon_clips_dqa_glued_sv_vehicle_day_FRONT_RIGHT_GENERIC_1x.mp4"
    
    # Test dummy inference
    tester.test_dummy_inference(video_path, max_frames=5)
    
    # Test trained model (if available)
    if os.path.exists("checkpoints/best_model.pth"):
        tester.test_trained_model_inference(video_path, max_frames=5)
    
    # Test pre-trained model
    tester.test_pretrained_inference(video_path, encoder_name="vgg11", max_frames=5)

if __name__ == "__main__":
    main()
