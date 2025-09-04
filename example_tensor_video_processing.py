#!/usr/bin/env python3
"""
Example script demonstrating video processing with tensor output for deep learning

This script shows how to use the VideoFrameIterator with tensor output,
perfect for feeding frames into neural networks like U-Net.
"""

import sys
from pathlib import Path

try:
    from video_processing import VideoFrameIterator, TensorFrameBatcher
    print("✓ Video processing module imported")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

try:
    import torch
    import torch.nn.functional as F
    print("✓ PyTorch available for tensor operations")
    TORCH_AVAILABLE = True
except ImportError:
    print("✗ PyTorch not available - install with: pip install torch")
    TORCH_AVAILABLE = False

try:
    import numpy as np
    print("✓ NumPy available")
except ImportError:
    print("✗ NumPy not available")


def example_deep_learning_preprocessing(video_path: str):
    """
    Example: Preprocess video frames for deep learning models
    """
    print(f"\n=== Deep Learning Preprocessing Example ===")
    print(f"Processing: {video_path}")
    
    try:
        # Process frames as normalized tensors ready for neural networks
        with VideoFrameIterator(
            video_path,
            frame_skip=5,           # Sample every 5th frame
            output_format="tensor",  # Output as PyTorch tensors
            normalize=True,         # Normalize to [0, 1] range
            resize_frames=(224, 224), # Standard input size for many models
            device="cpu",           # Use "cuda" if GPU available
            collect_frame_stats=True
        ) as iterator:
            
            processed_frames = []
            
            for i, (frame_tensor, metadata) in enumerate(iterator):
                # Frame tensor is ready for neural network input
                # Shape: (3, 224, 224) - (C, H, W) format
                # Values: [0, 1] range, RGB channel order
                
                print(f"Frame {metadata.frame_number}: "
                      f"Tensor {frame_tensor.shape}, "
                      f"Range [{frame_tensor.min():.3f}, {frame_tensor.max():.3f}]")
                
                # Example: Apply some tensor operations
                # Add batch dimension for model input
                frame_batch = frame_tensor.unsqueeze(0)  # Shape: (1, 3, 224, 224)
                
                # Example transformations you might apply
                if TORCH_AVAILABLE:
                    # Gaussian blur
                    blurred = F.gaussian_blur(frame_batch, kernel_size=3)
                    
                    # Edge detection using Sobel-like filter
                    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
                    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
                    
                    # Convert to grayscale for edge detection
                    gray = 0.299 * frame_batch[:, 0:1] + 0.587 * frame_batch[:, 1:2] + 0.114 * frame_batch[:, 2:3]
                    edges_x = F.conv2d(gray, sobel_x, padding=1)
                    edges_y = F.conv2d(gray, sobel_y, padding=1)
                    edges = torch.sqrt(edges_x**2 + edges_y**2)
                    
                    print(f"  Processed - Blur range: [{blurred.min():.3f}, {blurred.max():.3f}], "
                          f"Edge strength: {edges.mean():.4f}")
                
                processed_frames.append(frame_tensor)
                
                # Process only 10 frames for demo
                if i >= 9:
                    break
            
            print(f"✓ Processed {len(processed_frames)} frames as tensors")
            
            # Example: Stack all frames into a batch for model input
            if processed_frames:
                video_batch = torch.stack(processed_frames)  # Shape: (N, 3, 224, 224)
                print(f"✓ Created video batch: {video_batch.shape}")
                
                # This batch is now ready to feed into a neural network!
                # Example: model_output = unet_model(video_batch)
    
    except Exception as e:
        print(f"✗ Error during deep learning preprocessing: {e}")


def example_batch_tensor_processing(video_path: str):
    """
    Example: Process video frames in batches for efficient neural network inference
    """
    print(f"\n=== Batch Tensor Processing Example ===")
    
    try:
        batcher = TensorFrameBatcher(batch_size=8, device="cpu")
        batch_count = 0
        
        with VideoFrameIterator(
            video_path,
            frame_skip=10,
            output_format="tensor",
            normalize=True,
            resize_frames=(256, 256),
            device="cpu"
        ) as iterator:
            
            for frame_tensor, metadata in iterator:
                # Add frame to batch
                batch_tensor, batch_metadata = batcher.add_frame(frame_tensor, metadata)
                
                if batch_tensor is not None:
                    batch_count += 1
                    print(f"Batch {batch_count}: Shape {batch_tensor.shape}")
                    
                    # Here you would run your model inference
                    # Example: predictions = model(batch_tensor)
                    
                    # Example tensor operations on the batch
                    if TORCH_AVAILABLE:
                        batch_mean = batch_tensor.mean(dim=[2, 3])  # Mean across spatial dimensions
                        batch_std = batch_tensor.std(dim=[2, 3])    # Std across spatial dimensions
                        
                        print(f"  Batch statistics - Mean: {batch_mean.mean():.3f}, Std: {batch_std.mean():.3f}")
                    
                    if batch_count >= 3:  # Process 3 batches for demo
                        break
            
            # Process any remaining frames
            remaining_batch, remaining_metadata = batcher.get_remaining()
            if remaining_batch is not None:
                print(f"Final batch: Shape {remaining_batch.shape}")
        
        print(f"✓ Processed {batch_count} complete batches")
    
    except Exception as e:
        print(f"✗ Error during batch processing: {e}")


def example_unet_integration(video_path: str):
    """
    Example: How to integrate with U-Net for video segmentation
    """
    print(f"\n=== U-Net Integration Example ===")
    
    try:
        # This shows how you would process video frames for segmentation
        with VideoFrameIterator(
            video_path,
            frame_skip=30,          # Sample frames for segmentation
            output_format="tensor",  # Tensor output for model
            normalize=True,         # Normalize for neural network
            resize_frames=(512, 512), # U-Net input size
            device="cpu"            # Use "cuda" if available
        ) as iterator:
            
            segmentation_results = []
            
            for i, (frame_tensor, metadata) in enumerate(iterator):
                # Add batch dimension for U-Net input
                input_tensor = frame_tensor.unsqueeze(0)  # Shape: (1, 3, 512, 512)
                
                print(f"Frame {metadata.frame_number}: Ready for U-Net input {input_tensor.shape}")
                
                # Here's where you would run U-Net inference:
                # with torch.no_grad():
                #     segmentation_mask = unet_model(input_tensor)
                #     predictions = torch.argmax(segmentation_mask, dim=1)
                
                # For demo, we'll simulate the output
                if TORCH_AVAILABLE:
                    # Simulate segmentation output (21 classes for Pascal VOC)
                    mock_segmentation = torch.randint(0, 21, (1, 512, 512))
                    segmentation_results.append({
                        'frame_number': metadata.frame_number,
                        'timestamp': metadata.timestamp,
                        'segmentation_shape': mock_segmentation.shape,
                        'unique_classes': torch.unique(mock_segmentation).tolist()
                    })
                
                if i >= 5:  # Process 5 frames for demo
                    break
            
            print(f"✓ Processed {len(segmentation_results)} frames for segmentation")
            for result in segmentation_results:
                print(f"  Frame {result['frame_number']}: "
                      f"{len(result['unique_classes'])} unique classes detected")
    
    except Exception as e:
        print(f"✗ Error during U-Net integration example: {e}")


def main():
    """Main example script"""
    print("=== Video Processing with Tensor Support Examples ===")
    
    # Check dependencies
    try:
        import cv2
        print("✓ OpenCV available")
    except ImportError:
        print("✗ OpenCV not available - install with: pip install opencv-python")
        return
    
    try:
        import torch
        print("✓ PyTorch available")
    except ImportError:
        print("✗ PyTorch not available - install with: pip install torch")
        return
    
    # Create test video
    try:
        from test_video_iterator import create_test_video
        test_video_path = create_test_video("tensor_test_video.mp4", duration_seconds=3)
        
        if test_video_path and Path(test_video_path).exists():
            print(f"✓ Created test video: {test_video_path}")
            
            # Run examples
            example_deep_learning_preprocessing(test_video_path)
            example_batch_tensor_processing(test_video_path)
            example_unet_integration(test_video_path)
            
            print(f"\n=== Examples Complete ===")
            print("The video iterator is ready for your deep learning projects!")
            
            # Clean up
            Path(test_video_path).unlink()
            print("✓ Cleaned up test video")
        
        else:
            print("✗ Could not create test video")
    
    except Exception as e:
        print(f"✗ Error: {e}")


if __name__ == "__main__":
    main()
