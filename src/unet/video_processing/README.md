# Video Processing Module

A comprehensive video frame iterator with tensor support for deep learning applications.

## Features

- **Frame-by-frame iteration** with rich metadata collection
- **Tensor output support** for PyTorch models
- **Flexible frame sampling** (skip frames, time ranges)
- **Batch processing** for efficient model inference
- **Memory-efficient** processing with context managers
- **Deep learning ready** with normalized tensors in CHW format

## Quick Start

### Basic Usage
```python
from video_processing import VideoFrameIterator

# Process as numpy arrays (default)
with VideoFrameIterator("video.mp4") as iterator:
    for frame, metadata in iterator:
        print(f"Frame {metadata.frame_number}: {frame.shape}")
```

### Tensor Output for Deep Learning
```python
# Process as PyTorch tensors
with VideoFrameIterator(
    "video.mp4",
    output_format="tensor",
    normalize=True,
    resize_frames=(224, 224),
    device="cpu"  # or "cuda"
) as iterator:
    
    for frame_tensor, metadata in iterator:
        # frame_tensor is ready for neural network input
        # Shape: (3, 224, 224), Range: [0, 1], Format: RGB
        model_input = frame_tensor.unsqueeze(0)  # Add batch dim
        # predictions = model(model_input)
```

### Batch Processing
```python
from video_processing import TensorFrameBatcher

batcher = TensorFrameBatcher(batch_size=8, device="cpu")

with VideoFrameIterator("video.mp4", output_format="tensor") as iterator:
    for frame_tensor, metadata in iterator:
        batch_tensor, batch_metadata = batcher.add_frame(frame_tensor, metadata)
        
        if batch_tensor is not None:
            # Process batch with your model
            # predictions = model(batch_tensor)  # Shape: (8, 3, H, W)
            print(f"Processing batch: {batch_tensor.shape}")
```

### U-Net Integration Example
```python
# Perfect for semantic segmentation
with VideoFrameIterator(
    "video.mp4",
    output_format="tensor",
    normalize=True,
    resize_frames=(512, 512),  # U-Net input size
    frame_skip=10
) as iterator:
    
    for frame_tensor, metadata in iterator:
        # Ready for U-Net input
        input_batch = frame_tensor.unsqueeze(0)  # (1, 3, 512, 512)
        
        # with torch.no_grad():
        #     segmentation = unet_model(input_batch)
        #     predicted_mask = torch.argmax(segmentation, dim=1)
        
        print(f"Frame {metadata.frame_number} ready for segmentation")
```

## Output Formats

- **`"numpy"`** (default): Returns numpy arrays (H, W, C) in BGR format
- **`"tensor"`**: Returns PyTorch tensors (C, H, W) in RGB format, normalized
- **`"both"`**: Returns tuple of (numpy_array, tensor)

## Metadata Collected

Each frame comes with rich metadata:
- Frame number and timestamp
- Video information (fps, dimensions, codec)
- Frame statistics (brightness, color distribution)
- Tensor statistics (when using tensor output)
- Processing time
- Custom metadata via callbacks

## Installation

```bash
# Basic functionality
pip install opencv-python numpy

# For tensor support
pip install torch

# Full feature set
pip install opencv-python numpy torch
```

## Command Line Usage

```bash
# Basic processing
python3 -m video_processing.frame_iterator video.mp4

# Tensor output
python3 -m video_processing.frame_iterator video.mp4 --output_format tensor --normalize --resize 224 224

# Demo modes
python3 -m video_processing.frame_iterator video.mp4 --demo
python3 -m video_processing.frame_iterator video.mp4 --demo_tensor
```

This module is designed to seamlessly integrate with the U-Net codebase for video-based semantic segmentation tasks.
