#!/usr/bin/env python3
"""
Demo scripts for U-Net project

This module consolidates all example scripts:
- Instance integration demo
- Tensor video processing demo
- U-Net video segmentation demo
- Inference examples
- Training examples
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

from video_processing.frame_iterator import VideoFrameIterator, TensorFrameBatcher
from instance_extraction import InstanceExtractor
from data.bdd100k_dataset import BDD100KSegmentationDataset
from data.dataset_utils import create_sample_dataset_for_testing


class DemoScripts:
    """Consolidated demo scripts for U-Net project"""
    
    def __init__(self, output_base_dir: str = "demo_outputs"):
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
    
    def demo_instance_integration(
        self,
        video_path: str,
        output_dir: str = "instance_integration_demo",
        frame_skip: int = 30,
        max_frames: int = 5
    ) -> Dict[str, Any]:
        """Demo: Complete semantic to instance segmentation pipeline"""
        
        print("🔗 Instance Integration Demo")
        print("=" * 40)
        print("This demo shows the complete pipeline from semantic to instance segmentation")
        
        output_path = self.output_base_dir / output_dir
        output_path.mkdir(exist_ok=True)
        
        try:
            # Initialize instance extractor
            extractor = InstanceExtractor(algorithm="watershed")
            
            # Process video frames
            iterator = VideoFrameIterator(video_path, frame_skip=frame_skip, max_frames=max_frames)
            frame_count = 0
            total_instances = 0
            
            print(f"Processing video: {video_path}")
            print(f"Output directory: {output_path}")
            
            for frame, metadata in iterator:
                frame_count += 1
                print(f"\nProcessing frame {metadata.frame_number} ({frame_count}/{max_frames})...")
                
                # Step 1: Create dummy semantic segmentation
                print("  Step 1: Creating semantic segmentation...")
                semantic_mask = self._create_driving_scene_mask(frame)
                
                # Step 2: Extract instances
                print("  Step 2: Extracting instances...")
                instances = extractor.extract_instances(
                    semantic_mask=semantic_mask,
                    target_classes=[2, 11, 13],  # buildings, people, cars
                    min_instance_size=50
                )
                
                total_instances += len(instances['class_mapping'])
                print(f"    Found {len(instances['class_mapping'])} instances")
                
                # Step 3: Create visualizations
                print("  Step 3: Creating visualizations...")
                
                # Semantic visualization
                semantic_vis = self._create_semantic_visualization(semantic_mask)
                cv2.imwrite(str(output_path / f"frame_{metadata.frame_number:06d}_semantic.jpg"), semantic_vis)
                
                # Instance visualization
                instance_vis = extractor.visualize_instances(
                    instances=instances,
                    original_image=frame,
                    output_path=output_path / f"frame_{metadata.frame_number:06d}_instances.jpg",
                    show_overlay=True,
                    show_contours=True,
                    show_labels=True
                )
                
                # Save original frame
                cv2.imwrite(str(output_path / f"frame_{metadata.frame_number:06d}_original.jpg"), frame)
                
                print(f"    Saved visualizations for frame {metadata.frame_number}")
            
            results = {
                'success': True,
                'frames_processed': frame_count,
                'total_instances': total_instances,
                'avg_instances_per_frame': total_instances / frame_count if frame_count > 0 else 0,
                'output_dir': str(output_path)
            }
            
            print(f"\n✅ Instance integration demo completed!")
            print(f"   Processed {frame_count} frames")
            print(f"   Found {total_instances} total instances")
            print(f"   Average {results['avg_instances_per_frame']:.1f} instances per frame")
            print(f"   Results saved to: {output_path}")
            
        except Exception as e:
            print(f"❌ Instance integration demo failed: {e}")
            results = {'success': False, 'error': str(e)}
        
        return results
    
    def demo_tensor_video_processing(
        self,
        video_path: str,
        output_dir: str = "tensor_processing_demo",
        batch_size: int = 4,
        target_size: tuple = (256, 256)
    ) -> Dict[str, Any]:
        """Demo: Deep learning preprocessing with tensor batching"""
        
        print("🧠 Tensor Video Processing Demo")
        print("=" * 40)
        print("This demo shows deep learning preprocessing with tensor batching")
        
        output_path = self.output_base_dir / output_dir
        output_path.mkdir(exist_ok=True)
        
        try:
            # Initialize tensor batcher
            batcher = TensorFrameBatcher(batch_size=batch_size, target_size=target_size)
            
            # Process video
            iterator = VideoFrameIterator(video_path, frame_skip=20, max_frames=12)
            batch_count = 0
            total_frames = 0
            
            print(f"Processing video: {video_path}")
            print(f"Batch size: {batch_size}")
            print(f"Target size: {target_size}")
            
            for frame, metadata in iterator:
                total_frames += 1
                print(f"Processing frame {metadata.frame_number}...")
                
                # Add frame to batch
                batch_tensor = batcher.add_frame(frame, metadata)
                
                if batch_tensor is not None:
                    batch_count += 1
                    print(f"  Created batch {batch_count}: {batch_tensor.shape}")
                    
                    # Save batch visualization
                    self._save_batch_visualization(batch_tensor, batch_count, output_path)
            
            # Process any remaining frames
            final_batch = batcher.get_remaining_batch()
            if final_batch is not None:
                batch_count += 1
                print(f"  Created final batch {batch_count}: {final_batch.shape}")
                self._save_batch_visualization(final_batch, batch_count, output_path)
            
            results = {
                'success': True,
                'total_frames': total_frames,
                'batches_created': batch_count,
                'batch_size': batch_size,
                'output_dir': str(output_path)
            }
            
            print(f"\n✅ Tensor processing demo completed!")
            print(f"   Processed {total_frames} frames")
            print(f"   Created {batch_count} batches")
            print(f"   Results saved to: {output_path}")
            
        except Exception as e:
            print(f"❌ Tensor processing demo failed: {e}")
            results = {'success': False, 'error': str(e)}
        
        return results
    
    def demo_unet_video_segmentation(
        self,
        video_path: str,
        output_dir: str = "unet_segmentation_demo",
        frame_skip: int = 30,
        max_frames: int = 8
    ) -> Dict[str, Any]:
        """Demo: U-Net video segmentation with batch processing"""
        
        print("🎯 U-Net Video Segmentation Demo")
        print("=" * 40)
        print("This demo shows U-Net video segmentation with batch processing")
        
        output_path = self.output_base_dir / output_dir
        output_path.mkdir(exist_ok=True)
        
        try:
            # Create dummy U-Net model
            model = self._create_dummy_unet()
            
            # Process video frames
            iterator = VideoFrameIterator(video_path, frame_skip=frame_skip, max_frames=max_frames)
            frame_count = 0
            start_time = time.time()
            
            print(f"Processing video: {video_path}")
            print(f"Model: Dummy U-Net (simulated)")
            
            for frame, metadata in iterator:
                frame_count += 1
                print(f"Processing frame {metadata.frame_number} ({frame_count}/{max_frames})...")
                
                # Simulate U-Net inference
                prediction = self._simulate_unet_inference(frame)
                
                # Create visualization
                vis_image = self._create_segmentation_visualization(frame, prediction)
                
                # Save results
                base_name = f"frame_{metadata.frame_number:06d}"
                cv2.imwrite(str(output_path / f"{base_name}_original.jpg"), frame)
                cv2.imwrite(str(output_path / f"{base_name}_segmentation.jpg"), vis_image)
                
                # Create overlay
                overlay = cv2.addWeighted(frame, 0.7, vis_image, 0.3, 0)
                cv2.imwrite(str(output_path / f"{base_name}_overlay.jpg"), overlay)
                
                print(f"  Saved segmentation for frame {metadata.frame_number}")
            
            processing_time = time.time() - start_time
            fps = frame_count / processing_time if processing_time > 0 else 0
            
            results = {
                'success': True,
                'frames_processed': frame_count,
                'processing_time': processing_time,
                'fps': fps,
                'output_dir': str(output_path)
            }
            
            print(f"\n✅ U-Net segmentation demo completed!")
            print(f"   Processed {frame_count} frames in {processing_time:.2f} seconds")
            print(f"   Average FPS: {fps:.2f}")
            print(f"   Results saved to: {output_path}")
            
        except Exception as e:
            print(f"❌ U-Net segmentation demo failed: {e}")
            results = {'success': False, 'error': str(e)}
        
        return results
    
    def demo_inference_example(self, image_path: str = None) -> Dict[str, Any]:
        """Demo: Basic inference example"""
        
        print("🔍 Inference Example Demo")
        print("=" * 40)
        print("This demo shows basic inference functionality")
        
        try:
            if image_path is None:
                # Create a test image
                image_path = self._create_test_image()
                print(f"Created test image: {image_path}")
            
            # Load and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            print(f"Loaded image: {image.shape}")
            
            # Simulate inference
            prediction = self._simulate_inference(image)
            print(f"Generated prediction: {prediction.shape}")
            
            # Create visualization
            vis_image = self._create_segmentation_visualization(image, prediction)
            
            # Save results
            output_path = self.output_base_dir / "inference_demo"
            output_path.mkdir(exist_ok=True)
            
            cv2.imwrite(str(output_path / "original.jpg"), image)
            cv2.imwrite(str(output_path / "prediction.jpg"), vis_image)
            
            results = {
                'success': True,
                'image_path': image_path,
                'output_dir': str(output_path)
            }
            
            print(f"✅ Inference demo completed!")
            print(f"   Results saved to: {output_path}")
            
        except Exception as e:
            print(f"❌ Inference demo failed: {e}")
            results = {'success': False, 'error': str(e)}
        
        return results
    
    def demo_training_example(self, num_epochs: int = 3) -> Dict[str, Any]:
        """Demo: Basic training example"""
        
        print("🚀 Training Example Demo")
        print("=" * 40)
        print("This demo shows basic training functionality")
        
        try:
            # Create sample dataset
            print("Creating sample dataset...")
            sample_data = create_sample_dataset_for_testing(output_dir='data')
            print(f"Sample dataset created: {sample_data}")
            
            # Create simple model
            print("Creating model...")
            model = self._create_dummy_unet()
            print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
            
            # Simulate training
            print(f"Simulating training for {num_epochs} epochs...")
            start_time = time.time()
            
            for epoch in range(num_epochs):
                # Simulate training step
                time.sleep(0.1)  # Simulate processing time
                loss = 1.0 - (epoch + 1) * 0.2  # Simulate decreasing loss
                print(f"  Epoch {epoch+1}/{num_epochs}: Loss = {loss:.4f}")
            
            training_time = time.time() - start_time
            
            results = {
                'success': True,
                'epochs': num_epochs,
                'training_time': training_time,
                'final_loss': 1.0 - num_epochs * 0.2
            }
            
            print(f"✅ Training demo completed!")
            print(f"   Trained for {num_epochs} epochs in {training_time:.2f} seconds")
            print(f"   Final loss: {results['final_loss']:.4f}")
            
        except Exception as e:
            print(f"❌ Training demo failed: {e}")
            results = {'success': False, 'error': str(e)}
        
        return results
    
    def run_all_demos(self, video_path: str = None) -> Dict[str, Any]:
        """Run all demo scripts"""
        
        print("🎬 Running All Demo Scripts")
        print("=" * 50)
        
        if video_path is None:
            video_path = "video_processing/data3_users_yoavnavon_clips_dqa_glued_sv_vehicle_day_FRONT_RIGHT_GENERIC_1x.mp4"
        
        if not os.path.exists(video_path):
            print(f"⚠️ Video file not found: {video_path}")
            print("   Some demos will be skipped")
            video_path = None
        
        results = {}
        
        # Run demos
        if video_path:
            results['instance_integration'] = self.demo_instance_integration(video_path)
            results['tensor_processing'] = self.demo_tensor_video_processing(video_path)
            results['unet_segmentation'] = self.demo_unet_video_segmentation(video_path)
        
        results['inference'] = self.demo_inference_example()
        results['training'] = self.demo_training_example()
        
        # Print summary
        print(f"\n📋 Demo Summary")
        print("=" * 50)
        for demo_name, demo_results in results.items():
            status = "✅" if demo_results.get('success', False) else "❌"
            print(f"{demo_name}: {status}")
        
        return results
    
    # Helper methods
    
    def _create_driving_scene_mask(self, frame: np.ndarray) -> np.ndarray:
        """Create a realistic driving scene semantic mask"""
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Sky (top portion)
        mask[:h//3, :] = 10  # sky
        
        # Road (bottom portion)
        mask[2*h//3:, :] = 0  # road
        
        # Add cars
        for _ in range(np.random.randint(1, 4)):
            car_h = np.random.randint(20, 40)
            car_w = np.random.randint(30, 60)
            car_y = np.random.randint(h//2, h - car_h)
            car_x = np.random.randint(0, w - car_w)
            mask[car_y:car_y+car_h, car_x:car_x+car_w] = 13  # car
        
        # Add buildings
        for _ in range(np.random.randint(1, 3)):
            building_h = np.random.randint(40, h//2)
            building_w = np.random.randint(50, 100)
            building_y = np.random.randint(h//3, h//2)
            building_x = np.random.randint(0, w - building_w)
            mask[building_y:building_y+building_h, building_x:building_x+building_w] = 2  # building
        
        return mask
    
    def _create_semantic_visualization(self, semantic_mask: np.ndarray) -> np.ndarray:
        """Create colored visualization of semantic mask"""
        h, w = semantic_mask.shape
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Class colors
        colors = [
            (0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 0, 128),
            (255, 165, 0), (0, 128, 0), (128, 128, 128), (255, 192, 203),
            (165, 42, 42), (0, 0, 128), (128, 128, 0), (0, 128, 128),
            (128, 0, 0), (192, 192, 192), (255, 255, 255), (0, 0, 0),
            (64, 64, 64)
        ]
        
        for class_id, color in enumerate(colors):
            if class_id < len(colors):
                mask = semantic_mask == class_id
                vis[mask] = color
        
        return vis
    
    def _save_batch_visualization(self, batch_tensor: torch.Tensor, batch_num: int, output_dir: Path):
        """Save visualization of batch tensor"""
        batch_size = batch_tensor.shape[0]
        
        # Create grid visualization
        grid_size = int(np.ceil(np.sqrt(batch_size)))
        h, w = batch_tensor.shape[2], batch_tensor.shape[3]
        
        grid_image = np.zeros((grid_size * h, grid_size * w, 3), dtype=np.uint8)
        
        for i in range(batch_size):
            row = i // grid_size
            col = i % grid_size
            
            # Convert tensor to image
            img = batch_tensor[i].permute(1, 2, 0).numpy()
            img = (img * 255).astype(np.uint8)
            
            # Place in grid
            y_start = row * h
            y_end = y_start + h
            x_start = col * w
            x_end = x_start + w
            
            grid_image[y_start:y_end, x_start:x_end] = img
        
        # Save grid
        cv2.imwrite(str(output_dir / f"batch_{batch_num:03d}.jpg"), grid_image)
    
    def _create_dummy_unet(self):
        """Create a dummy U-Net model for demo purposes"""
        class DummyUNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 21, kernel_size=1)
            
            def forward(self, x):
                return self.conv(x)
        
        return DummyUNet()
    
    def _simulate_unet_inference(self, frame: np.ndarray) -> np.ndarray:
        """Simulate U-Net inference"""
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Create some dummy segmentation
        # Sky
        mask[:h//3, :] = 10
        
        # Road
        mask[2*h//3:, :] = 0
        
        # Add some objects
        for _ in range(3):
            x = np.random.randint(0, w-50)
            y = np.random.randint(h//3, 2*h//3)
            cv2.circle(mask, (x, y), 20, 6, -1)
        
        return mask
    
    def _simulate_inference(self, image: np.ndarray) -> np.ndarray:
        """Simulate inference on single image"""
        h, w = image.shape[:2]
        return self._simulate_unet_inference(image)
    
    def _create_segmentation_visualization(self, frame: np.ndarray, prediction: np.ndarray) -> np.ndarray:
        """Create segmentation visualization"""
        h, w = frame.shape[:2]
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Class colors
        colors = [
            (0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 0, 128),
            (255, 165, 0), (0, 128, 0), (128, 128, 128), (255, 192, 203),
            (165, 42, 42), (0, 0, 128), (128, 128, 0), (0, 128, 128),
            (128, 0, 0), (192, 192, 192), (255, 255, 255), (0, 0, 0),
            (64, 64, 64)
        ]
        
        for class_id, color in enumerate(colors):
            if class_id < len(colors):
                mask = prediction == class_id
                vis[mask] = color
        
        return vis
    
    def _create_test_image(self) -> str:
        """Create a test image for demo purposes"""
        # Create a simple test image
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # Add some structure
        img[:85, :] = [135, 206, 235]  # Sky
        img[170:, :] = [105, 105, 105]  # Road
        
        # Save image
        output_path = self.output_base_dir / "test_image.jpg"
        output_path.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(output_path), img)
        
        return str(output_path)


def main():
    """Run demo scripts"""
    demos = DemoScripts()
    
    # Check for video file
    video_path = "video_processing/data3_users_yoavnavon_clips_dqa_glued_sv_vehicle_day_FRONT_RIGHT_GENERIC_1x.mp4"
    if not os.path.exists(video_path):
        print(f"⚠️ Video file not found: {video_path}")
        print("   Running demos without video...")
        video_path = None
    
    results = demos.run_all_demos(video_path)
    
    # Save results
    results_file = demos.output_base_dir / "demo_results.txt"
    with open(results_file, 'w') as f:
        f.write("Demo Script Results\n")
        f.write("=" * 30 + "\n\n")
        
        for demo_name, demo_results in results.items():
            f.write(f"{demo_name.upper()}:\n")
            for key, value in demo_results.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
    
    print(f"\n📄 Results saved to: {results_file}")


if __name__ == "__main__":
    main()
