#!/usr/bin/env python3
"""
Inference testing module for U-Net project

This module consolidates tests for:
- Video inference with different model types
- Trained model inference
- Instance extraction
- Simple inference testing
"""

import sys
import os
import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path
import time
from typing import Dict, Any, Optional, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from video_processing.frame_iterator import VideoFrameIterator
from instance_extraction import InstanceExtractor


class InferenceTester:
    """Test inference functionality with different model types"""
    
    def __init__(self, output_base_dir: str = "test_outputs/inference_tests"):
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
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
        """Test trained model inference on video"""
        
        print("🧠 Testing Trained Model Inference")
        print("=" * 40)
        
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
    
    def test_pretrained_model_inference(
        self,
        video_path: str,
        encoder_name: str = "vgg11",
        output_dir: str = "pretrained_inference",
        frame_skip: int = 30,
        max_frames: int = 8
    ) -> Dict[str, Any]:
        """Test pre-trained model inference on video"""
        
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
    
    def test_instance_extraction(
        self,
        video_path: str,
        algorithm: str = "watershed",
        output_dir: str = "instance_extraction",
        frame_skip: int = 30,
        max_frames: int = 5
    ) -> Dict[str, Any]:
        """Test instance extraction on video"""
        
        print(f"🔍 Testing Instance Extraction ({algorithm})")
        print("=" * 40)
        
        output_path = self.output_base_dir / output_dir
        output_path.mkdir(exist_ok=True)
        
        try:
            # Initialize instance extractor
            extractor = InstanceExtractor(algorithm=algorithm)
            
            # Process video frames
            iterator = VideoFrameIterator(video_path, frame_skip=frame_skip, max_frames=max_frames)
            frame_count = 0
            total_instances = 0
            
            for frame, metadata in iterator:
                frame_count += 1
                print(f"Processing frame {metadata.frame_number} ({frame_count}/{max_frames})...")
                
                # Create dummy semantic mask for testing
                semantic_mask = self._create_dummy_semantic_mask(frame)
                
                # Extract instances
                instances = extractor.extract_instances(
                    semantic_mask=semantic_mask,
                    target_classes=[2, 11, 13],  # buildings, people, cars
                    min_instance_size=50
                )
                
                total_instances += len(instances['class_mapping'])
                
                # Visualize and save results
                vis_image = extractor.visualize_instances(
                    instances=instances,
                    original_image=frame,
                    output_path=output_path / f"frame_{metadata.frame_number:06d}_instances.jpg",
                    show_overlay=True,
                    show_contours=True,
                    show_labels=True
                )
                
                # Save original frame
                cv2.imwrite(str(output_path / f"frame_{metadata.frame_number:06d}_original.jpg"), frame)
            
            results = {
                'success': True,
                'frames_processed': frame_count,
                'total_instances': total_instances,
                'avg_instances_per_frame': total_instances / frame_count if frame_count > 0 else 0,
                'output_dir': str(output_path)
            }
            
            print(f"✅ Instance extraction completed!")
            print(f"   Processed {frame_count} frames")
            print(f"   Found {total_instances} total instances")
            print(f"   Average {results['avg_instances_per_frame']:.1f} instances per frame")
            
        except Exception as e:
            print(f"❌ Instance extraction failed: {e}")
            results = {
                'success': False,
                'error': str(e),
                'frames_processed': 0,
                'total_instances': 0
            }
        
        return results
    
    def _run_inference(
        self,
        video_path: str,
        model_type: str,
        model_path: Optional[str] = None,
        encoder_name: Optional[str] = None,
        output_dir: str = "inference_output",
        frame_skip: int = 30,
        max_frames: int = 8
    ) -> Dict[str, Any]:
        """Run inference with specified model type"""
        
        try:
            # Load model
            model = self._load_model(model_type, model_path, encoder_name)
            if model is None:
                return {'success': False, 'error': 'Failed to load model'}
            
            # Process video
            iterator = VideoFrameIterator(video_path, frame_skip=frame_skip, max_frames=max_frames)
            frame_count = 0
            start_time = time.time()
            
            for frame, metadata in iterator:
                frame_count += 1
                print(f"Processing frame {metadata.frame_number} ({frame_count}/{max_frames})...")
                
                # Preprocess frame
                input_tensor = self._preprocess_frame(frame)
                
                # Run inference
                with torch.no_grad():
                    if model_type == "dummy":
                        prediction = self._create_dummy_prediction(frame)
                    else:
                        output = model(input_tensor)
                        prediction = torch.argmax(output, dim=1).squeeze().cpu().numpy()
                
                # Create visualization
                vis_image = self._create_visualization(frame, prediction)
                
                # Save results
                base_name = f"frame_{metadata.frame_number:06d}"
                cv2.imwrite(os.path.join(output_dir, f"{base_name}_original.jpg"), frame)
                cv2.imwrite(os.path.join(output_dir, f"{base_name}_mask.png"), prediction.astype(np.uint8) * 10)
                cv2.imwrite(os.path.join(output_dir, f"{base_name}_visualization.jpg"), vis_image)
                
                # Create overlay
                overlay = cv2.addWeighted(frame, 0.7, vis_image, 0.3, 0)
                cv2.imwrite(os.path.join(output_dir, f"{base_name}_overlay.jpg"), overlay)
            
            processing_time = time.time() - start_time
            fps = frame_count / processing_time if processing_time > 0 else 0
            
            results = {
                'success': True,
                'frames_processed': frame_count,
                'processing_time': processing_time,
                'fps': fps,
                'output_dir': output_dir
            }
            
            print(f"✅ {model_type.title()} inference completed!")
            print(f"   Processed {frame_count} frames in {processing_time:.2f} seconds")
            print(f"   Average FPS: {fps:.2f}")
            
        except Exception as e:
            print(f"❌ {model_type.title()} inference failed: {e}")
            results = {
                'success': False,
                'error': str(e),
                'frames_processed': 0
            }
        
        return results
    
    def _load_model(self, model_type: str, model_path: Optional[str] = None, encoder_name: Optional[str] = None):
        """Load model based on type"""
        
        if model_type == "dummy":
            return "dummy"  # Special case for dummy inference
        
        elif model_type == "trained":
            if model_path and os.path.exists(model_path):
                # Load trained U-Net model
                from models.unet import UNet
                model = UNet(in_channels=3, num_classes=21)
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
    
    def _create_dummy_prediction(self, frame: np.ndarray) -> np.ndarray:
        """Create dummy segmentation prediction"""
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Create some dummy regions
        # Sky (top portion)
        mask[:h//3, :] = 10
        
        # Road (bottom portion)
        mask[2*h//3:, :] = 0
        
        # Add some random objects
        for _ in range(3):
            x = np.random.randint(0, w-50)
            y = np.random.randint(h//3, 2*h//3)
            cv2.circle(mask, (x, y), 20, 6, -1)
        
        return mask
    
    def _create_dummy_semantic_mask(self, frame: np.ndarray) -> np.ndarray:
        """Create dummy semantic mask for instance extraction testing"""
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Sky
        mask[:h//3, :] = 10
        
        # Road
        mask[2*h//3:, :] = 0
        
        # Add some cars
        for i in range(3):
            car_h = np.random.randint(20, 40)
            car_w = np.random.randint(30, 60)
            car_y = np.random.randint(h//2, h - car_h)
            car_x = np.random.randint(0, w - car_w)
            mask[car_y:car_y+car_h, car_x:car_x+car_w] = 13
        
        # Add buildings
        for i in range(2):
            building_h = np.random.randint(40, 80)
            building_w = np.random.randint(50, 100)
            building_y = np.random.randint(0, h//2)
            building_x = np.random.randint(0, w - building_w)
            mask[building_y:building_y+building_h, building_x:building_x+building_w] = 2
        
        return mask
    
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
    
    def run_all_inference_tests(
        self, 
        video_path: str, 
        model_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run all inference tests"""
        
        print("🧪 Running All Inference Tests")
        print("=" * 50)
        
        results = {}
        
        # Test dummy inference
        results['dummy'] = self.test_dummy_inference(video_path)
        
        # Test trained model inference
        if model_path and os.path.exists(model_path):
            results['trained'] = self.test_trained_model_inference(video_path, model_path)
        else:
            print("⚠️ Skipping trained model test (no model found)")
            results['trained'] = {'success': False, 'error': 'No model found'}
        
        # Test pre-trained models
        for encoder in ['vgg11', 'efficientnet']:
            results[f'pretrained_{encoder}'] = self.test_pretrained_model_inference(
                video_path, encoder
            )
        
        # Test instance extraction
        for algorithm in ['watershed', 'connected_components']:
            results[f'instance_{algorithm}'] = self.test_instance_extraction(
                video_path, algorithm
            )
        
        # Print summary
        print(f"\n📋 Inference Test Summary")
        print("=" * 50)
        for test_name, test_results in results.items():
            status = "✅" if test_results.get('success', False) else "❌"
            print(f"{test_name}: {status}")
        
        return results


def main():
    """Run inference tests"""
    tester = InferenceTester()
    
    # Check for video file
    video_path = "video_processing/data3_users_yoavnavon_clips_dqa_glued_sv_vehicle_day_FRONT_RIGHT_GENERIC_1x.mp4"
    if not os.path.exists(video_path):
        print("❌ Video file not found. Please provide a valid video path.")
        return
    
    # Check for trained model
    model_path = "checkpoints/best_model.pth"
    if not os.path.exists(model_path):
        model_path = None
    
    results = tester.run_all_inference_tests(video_path, model_path)
    
    # Save results
    results_file = tester.output_base_dir / "inference_test_results.txt"
    with open(results_file, 'w') as f:
        f.write("Inference Test Results\n")
        f.write("=" * 30 + "\n\n")
        
        for test_name, test_results in results.items():
            f.write(f"{test_name.upper()}:\n")
            for key, value in test_results.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
    
    print(f"\n📄 Results saved to: {results_file}")


if __name__ == "__main__":
    main()
