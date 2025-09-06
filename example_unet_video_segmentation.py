#!/usr/bin/env python3
"""
Example: Video Semantic Segmentation with U-Net

This script demonstrates how to use the video processing module
with the U-Net model for video semantic segmentation.
"""

import sys
from pathlib import Path

try:
    from video_processing import VideoFrameIterator, TensorFrameBatcher
    print("✓ Video processing module available")
except ImportError as e:
    print(f"✗ Video processing import error: {e}")
    sys.exit(1)

try:
    import torch
    import torch.nn.functional as F
    print("✓ PyTorch available")
except ImportError:
    print("✗ PyTorch not available - install with: pip install torch")
    sys.exit(1)

try:
    from models import UNet, create_unet_model
    print("✓ U-Net model available")
    UNET_AVAILABLE = True
except ImportError:
    print("⚠️  U-Net model not available (missing dependencies)")
    UNET_AVAILABLE = False


def simulate_unet_video_segmentation(video_path: str, output_dir: str = "segmentation_output"):
    """
    Simulate video semantic segmentation using U-Net
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save segmentation results
    """
    print(f"\n=== Video Semantic Segmentation Demo ===")
    print(f"Input video: {video_path}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    try:
        # Initialize U-Net model (simulated if dependencies not available)
        if UNET_AVAILABLE:
            # Create actual U-Net model
            config = {
                'model': {
                    'input_channels': 3,
                    'num_classes': 21,  # Pascal VOC classes
                    'dropout': 0.0
                }
            }
            model = create_unet_model(config)
            model.eval()
            print("✓ U-Net model created and set to eval mode")
        else:
            model = None
            print("⚠️  Using simulated segmentation (U-Net not available)")
        
        segmentation_results = []
        
        # Process video frames for segmentation
        with VideoFrameIterator(
            video_path,
            frame_skip=15,          # Process every 15th frame
            output_format="tensor", # Tensor output for model
            normalize=True,         # Normalize to [0, 1]
            resize_frames=(512, 512), # U-Net input size
            device="cpu",           # Use "cuda" if available
            collect_frame_stats=True
        ) as iterator:
            
            for i, (frame_tensor, metadata) in enumerate(iterator):
                print(f"\nProcessing frame {metadata.frame_number} ({metadata.timestamp:.2f}s)")
                
                # Prepare input for U-Net
                input_tensor = frame_tensor.unsqueeze(0)  # Add batch dimension: (1, 3, 512, 512)
                
                if model is not None:
                    # Run actual U-Net inference
                    with torch.no_grad():
                        segmentation_logits = model(input_tensor)
                        segmentation_probs = F.softmax(segmentation_logits, dim=1)
                        predicted_mask = torch.argmax(segmentation_probs, dim=1)
                        
                        # Get prediction statistics
                        unique_classes = torch.unique(predicted_mask).tolist()
                        confidence_scores = segmentation_probs.max(dim=1)[0].mean().item()
                        
                        print(f"  ✓ Segmentation complete: {len(unique_classes)} classes detected")
                        print(f"  ✓ Average confidence: {confidence_scores:.3f}")
                
                else:
                    # Simulate segmentation output
                    predicted_mask = torch.randint(0, 21, (1, 512, 512))
                    unique_classes = torch.unique(predicted_mask).tolist()
                    confidence_scores = 0.85  # Simulated confidence
                    
                    print(f"  ⚠️  Simulated segmentation: {len(unique_classes)} classes")
                
                # Store results
                result = {
                    'frame_number': metadata.frame_number,
                    'timestamp': metadata.timestamp,
                    'unique_classes': unique_classes,
                    'confidence_score': confidence_scores,
                    'input_shape': list(input_tensor.shape),
                    'output_shape': list(predicted_mask.shape)
                }
                segmentation_results.append(result)
                
                # Process 5 frames for demo
                if i >= 4:
                    break
        
        print(f"\n✓ Processed {len(segmentation_results)} frames")
        
        # Save results
        import json
        results_file = output_path / "segmentation_results.json"
        with open(results_file, 'w') as f:
            json.dump(segmentation_results, f, indent=2)
        
        print(f"✓ Results saved to: {results_file}")
        
        # Print summary
        total_classes = set()
        avg_confidence = 0
        for result in segmentation_results:
            total_classes.update(result['unique_classes'])
            avg_confidence += result['confidence_score']
        
        avg_confidence /= len(segmentation_results)
        
        print(f"\n=== Segmentation Summary ===")
        print(f"Frames processed: {len(segmentation_results)}")
        print(f"Unique classes found: {sorted(total_classes)}")
        print(f"Average confidence: {avg_confidence:.3f}")
        
    except Exception as e:
        print(f"✗ Error during video segmentation: {e}")


def example_batch_segmentation(video_path: str):
    """
    Example: Batch processing for efficient video segmentation
    """
    print(f"\n=== Batch Video Segmentation Demo ===")
    
    try:
        batcher = TensorFrameBatcher(batch_size=4, device="cpu")
        batch_count = 0
        
        with VideoFrameIterator(
            video_path,
            frame_skip=20,
            output_format="tensor",
            normalize=True,
            resize_frames=(256, 256),  # Smaller for faster processing
            device="cpu"
        ) as iterator:
            
            for frame_tensor, metadata in iterator:
                batch_tensor, batch_metadata = batcher.add_frame(frame_tensor, metadata)
                
                if batch_tensor is not None:
                    batch_count += 1
                    print(f"\nBatch {batch_count}: Processing {batch_tensor.shape}")
                    
                    if UNET_AVAILABLE and model is not None:
                        # Process entire batch at once (more efficient)
                        with torch.no_grad():
                            batch_segmentation = model(batch_tensor)
                            batch_predictions = torch.argmax(batch_segmentation, dim=1)
                            
                            print(f"  ✓ Batch segmentation complete: {batch_predictions.shape}")
                    else:
                        print(f"  ⚠️  Simulated batch processing")
                    
                    # Print frame info from batch
                    for j, meta in enumerate(batch_metadata):
                        print(f"    Frame {meta.frame_number}: {meta.timestamp:.2f}s")
                    
                    if batch_count >= 2:  # Process 2 batches for demo
                        break
            
            # Process any remaining frames
            remaining_batch, remaining_metadata = batcher.get_remaining()
            if remaining_batch is not None:
                print(f"\nFinal batch: {remaining_batch.shape}")
        
        print(f"✓ Batch processing complete: {batch_count} full batches processed")
    
    except Exception as e:
        print(f"✗ Error during batch segmentation: {e}")


def main():
    """Main demo function"""
    print("=== U-Net Video Segmentation Integration Demo ===")
    
    # Create test video
    try:
        from test_video_iterator import create_test_video
        test_video_path = create_test_video("segmentation_test_video.mp4", duration_seconds=4)
        
        if test_video_path and Path(test_video_path).exists():
            print(f"✓ Created test video: {test_video_path}")
            
            # Run examples
            simulate_unet_video_segmentation(test_video_path)
            example_batch_segmentation(test_video_path)
            
            print(f"\n=== Integration Demo Complete ===")
            print("The video processing module is ready for your U-Net projects!")
            print("\nTo use with real U-Net:")
            print("1. Install dependencies: pip install torch torchvision opencv-python")
            print("2. Load your trained U-Net model")
            print("3. Replace simulation code with actual model inference")
            
            # Clean up
            Path(test_video_path).unlink()
            print("✓ Cleaned up test video")
        
        else:
            print("✗ Could not create test video")
    
    except Exception as e:
        print(f"✗ Error: {e}")


if __name__ == "__main__":
    main()
