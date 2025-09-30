# Video Inference Testing Guide

This guide shows you how to test inference on video files using your U-Net semantic segmentation model.

## Quick Start

### 1. Test with Dummy Segmentation (Demo)
```bash
python3 simple_video_inference_test.py
```

This will process your video with dummy segmentation to demonstrate the pipeline.

### 2. Test with Real U-Net Model
```bash
python3 test_video_inference.py \
    --video video_processing/data3_users_yoavnavon_clips_dqa_glued_sv_vehicle_day_FRONT_RIGHT_GENERIC_1x.mp4 \
    --model path/to/your/trained_model.pth \
    --config config/config.yaml \
    --frame-skip 10 \
    --max-frames 20
```

## What Gets Generated

For each processed frame, you'll get 4 output files:

1. **`*_original.jpg`** - Original video frame
2. **`*_mask.png`** - Segmentation mask (grayscale)
3. **`*_visualization.jpg`** - Colored segmentation overlay
4. **`*_overlay.jpg`** - Original frame + colored segmentation

## Parameters

- `--video`: Path to your input video file
- `--model`: Path to trained U-Net model (.pth file)
- `--config`: Path to configuration file (.yaml)
- `--output`: Output directory (default: "video_inference_output")
- `--frame-skip`: Process every Nth frame (default: 10)
- `--max-frames`: Maximum frames to process (default: 50)

## Example Commands

### Process every 5th frame, max 30 frames
```bash
python3 test_video_inference.py \
    --video your_video.mp4 \
    --frame-skip 5 \
    --max-frames 30
```

### Process with custom output directory
```bash
python3 test_video_inference.py \
    --video your_video.mp4 \
    --output my_results \
    --frame-skip 15
```

## Performance Notes

- **Frame Skip**: Higher values = faster processing, fewer frames
- **Resolution**: Original video resolution is preserved
- **Memory**: Large videos may require more memory
- **Speed**: Depends on model complexity and hardware

## Troubleshooting

### No Model Available
If you don't have a trained model yet, use the dummy test:
```bash
python3 simple_video_inference_test.py
```

### Import Errors
Make sure you're in the project root directory and all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Video Format Issues
The iterator supports common video formats (MP4, AVI, MOV). If you have issues, try converting your video to MP4.

## Integration with Training

To use this with your trained U-Net model:

1. Train your model first:
   ```bash
   python3 main.py train --config config/config.yaml
   ```

2. Run video inference:
   ```bash
   python3 test_video_inference.py \
       --video your_video.mp4 \
       --model checkpoints/best_model.pth \
       --config config/config.yaml
   ```

## Output Analysis

The generated files allow you to:
- **Visualize** segmentation results
- **Compare** original vs segmented frames
- **Analyze** model performance on video data
- **Create** video overlays for presentations

## Next Steps

1. Train your U-Net model on your dataset
2. Test inference on various video types
3. Adjust frame skipping based on your needs
4. Integrate with your main training pipeline
