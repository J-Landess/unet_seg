# Semantic Segmentation U-Net

A modular, extensible implementation of U-Net for semantic segmentation using PyTorch, OpenCV, and other computer vision libraries.

## Features

- **Modular Architecture**: Clean, extensible codebase with separate modules for models, data, training, inference, and utilities
- **U-Net Implementation**: Complete U-Net architecture with skip connections and customizable depth
- **Data Augmentation**: Built-in support for various augmentations using Albumentations
- **Comprehensive Metrics**: Pixel accuracy, mean IoU, Dice coefficient, and per-class metrics
- **Visualization Tools**: Rich visualization utilities for predictions, training curves, and analysis
- **Flexible Configuration**: YAML-based configuration system
- **Logging Support**: TensorBoard and Weights & Biases integration
- **Easy Inference**: Simple API for single image and batch inference

## Project Structure

```
semantic_segmentation_unet/
├── config/
│   └── config.yaml              # Configuration file
├── data/
│   ├── __init__.py
│   └── dataset.py               # Dataset classes and data loaders
├── models/
│   ├── __init__.py
│   └── unet.py                  # U-Net model implementation
├── training/
│   ├── __init__.py
│   └── trainer.py               # Training pipeline
├── inference/
│   ├── __init__.py
│   └── inference.py             # Inference engine
├── utils/
│   ├── __init__.py
│   ├── metrics.py               # Evaluation metrics
│   └── visualization.py         # Visualization utilities
├── main.py                      # Main entry point
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## Installation

1. **Clone or create the project directory**:
   ```bash
   mkdir semantic_segmentation_unet
   cd semantic_segmentation_unet
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup data structure**:
   ```bash
   python main.py setup
   ```

## Quick Start

### 1. Prepare Your Data

Organize your data in the following structure:
```
data/
├── train/
│   ├── images/          # Training images
│   └── masks/           # Training masks (same filenames as images)
├── val/
│   ├── images/          # Validation images
│   └── masks/           # Validation masks
└── test/
    ├── images/          # Test images
    └── masks/           # Test masks
```

### 2. Configure Training

Edit `config/config.yaml` to match your dataset:
```yaml
model:
  num_classes: 21        # Number of classes in your dataset
  input_channels: 3      # RGB images

data:
  train_dir: "data/train"
  val_dir: "data/val"
  image_size: [512, 512] # Input image size

training:
  batch_size: 8
  num_epochs: 100
  learning_rate: 0.001
```

### 3. Train the Model

```bash
python main.py train --config config/config.yaml
```

### 4. Run Inference

```bash
# Single image
python main.py infer --model checkpoints/best_model.pth --config config/config.yaml --input test_image.jpg --output outputs/

# Batch processing
python main.py infer --model checkpoints/best_model.pth --config config/config.yaml --input data/test/images/ --output outputs/ --batch
```

### 5. Evaluate the Model

```bash
python main.py evaluate --model checkpoints/best_model.pth --config config/config.yaml --dataset data/test/ --output evaluation_results/
```

## Usage Examples

### Training with Custom Configuration

```python
from training import SegmentationTrainer

# Initialize trainer
trainer = SegmentationTrainer("config/config.yaml")

# Start training
trainer.train()

# Resume from checkpoint
trainer.train(resume_from="checkpoints/checkpoint_epoch_50.pth")
```

### Inference on Custom Images

```python
from inference import SegmentationInference

# Initialize inference engine
inference = SegmentationInference(
    model_path="checkpoints/best_model.pth",
    config_path="config/config.yaml"
)

# Predict on single image
prediction, probabilities = inference.predict_single("image.jpg")

# Create visualization
visualization = inference.visualize_prediction("image.jpg", prediction)

# Save results
inference.save_prediction(prediction, "outputs/prediction.png")
```

### Custom Dataset

```python
from data import SegmentationDataset, create_dataloader

# Create custom dataset
dataset = SegmentationDataset(
    image_dir="path/to/images",
    mask_dir="path/to/masks",
    image_size=(512, 512),
    augmentation={
        'horizontal_flip': 0.5,
        'rotation': 15,
        'brightness_contrast': 0.2
    },
    is_training=True
)

# Create dataloader
dataloader = create_dataloader(dataset, batch_size=8, shuffle=True)
```

### Visualization

```python
from utils import SegmentationVisualizer

# Initialize visualizer
visualizer = SegmentationVisualizer()

# Create prediction visualization
fig = visualizer.visualize_prediction(
    image=original_image,
    prediction=prediction_mask,
    ground_truth=gt_mask  # optional
)

# Plot training curves
fig = visualizer.plot_training_curves(
    train_losses=train_losses,
    val_losses=val_losses,
    train_metrics=train_metrics,
    val_metrics=val_metrics
)
```

## Configuration

The `config/config.yaml` file contains all training and model parameters:

### Model Configuration
- `input_channels`: Number of input channels (3 for RGB)
- `num_classes`: Number of segmentation classes
- `encoder_depth`: Depth of U-Net encoder
- `dropout`: Dropout rate for regularization

### Training Configuration
- `batch_size`: Training batch size
- `num_epochs`: Number of training epochs
- `learning_rate`: Initial learning rate
- `weight_decay`: L2 regularization weight
- `scheduler`: Learning rate scheduler type
- `early_stopping_patience`: Early stopping patience

### Data Configuration
- `train_dir`/`val_dir`/`test_dir`: Dataset directories
- `image_size`: Input image size [height, width]
- `augmentation`: Data augmentation parameters

## Extending the Framework

### Adding New Models

1. Create a new model file in `models/`
2. Implement the model class with `forward()` method
3. Add to `models/__init__.py`
4. Update the trainer to use your model

### Adding New Datasets

1. Create a new dataset class inheriting from `SegmentationDataset`
2. Implement the required methods
3. Add to `data/__init__.py`

### Adding New Metrics

1. Add metric functions to `utils/metrics.py`
2. Update the `SegmentationMetrics` class
3. Add to `utils/__init__.py`

## Dependencies

- **PyTorch**: Deep learning framework
- **OpenCV**: Computer vision operations
- **Albumentations**: Data augmentation
- **Matplotlib/Seaborn**: Visualization
- **TensorBoard**: Training monitoring
- **Weights & Biases**: Experiment tracking (optional)

## Performance Tips

1. **Use GPU**: Ensure CUDA is available for faster training
2. **Batch Size**: Adjust batch size based on GPU memory
3. **Data Loading**: Use multiple workers for data loading
4. **Mixed Precision**: Consider using automatic mixed precision for faster training
5. **Model Optimization**: Use model compilation for inference speedup

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or image size
2. **Data Loading Errors**: Check file paths and formats
3. **Poor Performance**: Adjust learning rate, add more data augmentation
4. **Slow Training**: Use GPU, increase num_workers, enable mixed precision

### Getting Help

- Check the configuration file format
- Verify data directory structure
- Ensure all dependencies are installed
- Check GPU availability and CUDA installation

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## Acknowledgments

- U-Net paper: "U-Net: Convolutional Networks for Biomedical Image Segmentation"
- PyTorch team for the excellent deep learning framework
- Albumentations for comprehensive data augmentation
