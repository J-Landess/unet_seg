# U-Net Semantic Segmentation

A comprehensive, production-ready implementation of U-Net for semantic segmentation with support for BDD100K and KITTI datasets, video processing, instance extraction, and advanced deep learning features.

![Sample Segmentation](assets/sample_segmentation.jpg)
*Example semantic segmentation result on BDD100K dataset*

## 🚀 Features

- **Complete U-Net Architecture**: 31M parameters with skip connections and customizable depth
- **Real-World Datasets**: BDD100K (100K driving images) and KITTI support
- **Video Processing**: Frame-by-frame video analysis with tensor output
- **Instance Extraction**: Watershed and Connected Components algorithms for instance segmentation
- **Pre-trained Models**: VGG11, EfficientNet, and other architectures via segmentation-models-pytorch
- **Advanced Data Augmentation**: Albumentations with driving-specific augmentations
- **Professional Metrics**: Pixel accuracy, mean IoU, Dice coefficient, per-class metrics
- **Comprehensive Visualization**: Training curves, prediction overlays, dataset analysis
- **Flexible Configuration**: YAML-based configuration system
- **Experiment Tracking**: TensorBoard and Weights & Biases integration
- **Production Ready**: Complete CLI interface, error handling, logging
- **Comprehensive Testing**: Consolidated test suite with core, inference, and training tests

## 📦 Installation

### Option 1: Pip Install (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/unet-semantic-segmentation.git
cd unet-semantic-segmentation

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e .[dev]
```

### Option 2: Conda Environment

```bash
# Create environment
conda env create -f environment-macos.yml  # macOS
# or
conda env create -f environment-cuda.yml   # Linux/Windows with CUDA

# Activate environment
conda activate unet-semantic-segmentation
```

### Option 3: Manual Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🎯 Quick Start

### Command Line Interface

```bash
# Setup data structure
unet-setup

# Train model
unet-train --config config/config.yaml

# Run inference
unet-infer --model checkpoints/best_model.pth --input image.jpg --output outputs/

# Evaluate model
unet-evaluate --model checkpoints/best_model.pth --dataset data/test/ --output results/
```

### Python API

#### Training
```python
from unet import SegmentationTrainer

# Initialize trainer
trainer = SegmentationTrainer("config/config.yaml")

# Start training
trainer.train()

# Resume from checkpoint
trainer.train(resume_from="checkpoints/checkpoint_epoch_50.pth")
```

#### Inference
```python
from unet import SegmentationInference

# Initialize inference engine
inference = SegmentationInference(
    model_path="checkpoints/best_model.pth",
    config_path="config/config.yaml"
)

# Predict on single image
prediction, probabilities = inference.predict_single("image.jpg")

# Create visualization
visualization = inference.visualize_prediction("image.jpg", prediction)
```

#### Video Processing
```python
from unet import VideoFrameIterator, TensorFrameBatcher

# Process video frames
with VideoFrameIterator("video.mp4", output_format="tensor") as iterator:
    for frame, metadata in iterator:
        # frame is a PyTorch tensor ready for neural networks
        print(f"Frame {metadata.frame_number}: {frame.shape}")

# Batch processing
batcher = TensorFrameBatcher(batch_size=8)
# ... add frames to batcher
batch = batcher.get_batch()  # Ready for model inference
```

#### Instance Extraction
```python
from unet import InstanceExtractor

# Initialize extractor
extractor = InstanceExtractor(algorithm="watershed")

# Extract instances from semantic mask
instances = extractor.extract_instances(
    semantic_mask=semantic_mask,
    target_classes=[2, 11, 13],  # buildings, people, cars
    min_instance_size=50
)

# Visualize results
visualization = extractor.visualize_instances(
    instances=instances,
    original_image=image,
    show_overlay=True,
    show_contours=True
)
```

## 📊 Results & Examples

### Instance Segmentation
![Instance Segmentation Comparison](assets/comparison_plot.png)
*Comparison of Watershed vs Connected Components algorithms*

![Watershed Instances](assets/watershed_instances.jpg)
*Watershed algorithm results*

![Connected Components Instances](assets/connected_components_instances.jpg)
*Connected Components algorithm results*

### Semantic Segmentation
![Semantic Segmentation](assets/semantic_segmentation.jpg)
*Semantic segmentation on driving scene*

## 🧪 Testing

The project includes a comprehensive test suite:

```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/core_tests.py
python -m pytest tests/inference_tests.py
python -m pytest tests/training_tests.py

# Run with coverage
python -m pytest tests/ --cov=unet --cov-report=html
```

## 🔧 Configuration

Edit `config/config.yaml` to customize your setup:

```yaml
model:
  name: "unet"
  input_channels: 3
  num_classes: 19              # 19 for BDD100K/KITTI
  encoder_depth: 5
  dropout: 0.2

data:
  dataset_type: "bdd100k"      # "bdd100k" or "kitti"
  root_dir: "datasets/bdd100k"
  image_size: [512, 512]
  augmentation:
    horizontal_flip: 0.5
    brightness_contrast: 0.2
    shadow: 0.1
    fog: 0.05

training:
  batch_size: 8
  num_epochs: 100
  learning_rate: 0.001
  weight_decay: 1e-4
  scheduler: "cosine"
  early_stopping_patience: 10
```

## 📁 Project Structure

```
unet/
├── src/unet/                   # Main package
│   ├── __init__.py
│   ├── cli.py                  # Command line interface
│   ├── models/                 # Model implementations
│   ├── data/                   # Dataset classes
│   ├── training/               # Training pipeline
│   ├── inference/              # Inference engine
│   ├── instance_extraction/    # Instance segmentation
│   ├── video_processing/       # Video processing
│   ├── utils/                  # Utilities and metrics
│   ├── tests/                  # Test suite
│   └── examples/               # Example scripts
├── config/                     # Configuration files
├── assets/                     # Example images and results
├── checkpoints/                # Model checkpoints
├── data/                       # Data directories
├── outputs/                    # Output results
├── pyproject.toml              # Package configuration
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## 📊 Supported Datasets

### BDD100K Dataset
- **100K driving images** with 19 semantic classes
- **Weather augmentations**: fog, rain, shadow
- **Class balancing** with computed weights
- **Download**: https://bdd-data.berkeley.edu/

### KITTI Dataset
- **Autonomous driving scenes** with detailed labels
- **Sequence support** for video processing
- **Temporal analysis** capabilities
- **Download**: http://www.cvlibs.net/datasets/kitti/

### Custom Datasets
```python
from unet import SegmentationDataset

# Create custom dataset
dataset = SegmentationDataset(
    image_dir="path/to/images",
    mask_dir="path/to/masks",
    image_size=(512, 512),
    augmentation={
        'horizontal_flip': 0.5,
        'rotation': 15,
        'brightness_contrast': 0.2
    }
)
```

## 🏗️ Pre-trained Models

Use pre-trained models for better performance:

```python
# VGG11-based U-Net
from segmentation_models_pytorch import Unet

model = Unet(
    encoder_name="vgg11",
    encoder_weights="imagenet",
    classes=19,
    activation=None
)

# EfficientNet-based U-Net
model = Unet(
    encoder_name="efficientnet-b0",
    encoder_weights="imagenet",
    classes=19,
    activation=None
)
```

## 📈 Performance & Monitoring

### Metrics
- **Pixel Accuracy**: Overall pixel-level accuracy
- **Mean IoU**: Intersection over Union for each class
- **Dice Coefficient**: Overlap measure for segmentation
- **Per-Class Metrics**: Detailed analysis for each class
- **Instance Metrics**: Instance-level evaluation

### Visualization
- **Training Curves**: Loss and metric plots
- **Prediction Overlays**: Side-by-side comparisons
- **Confusion Matrices**: Class-wise performance
- **Dataset Analysis**: Class distribution and statistics
- **Instance Visualizations**: Contour and label overlays

### Experiment Tracking
- **TensorBoard**: Real-time training monitoring
- **Weights & Biases**: Advanced experiment tracking
- **Logging**: Comprehensive training logs

## 🔍 Troubleshooting

### Common Issues

1. **OpenCV Import Error (macOS)**
   ```bash
   # Use headless version
   pip install opencv-python-headless
   ```

2. **CUDA Out of Memory**
   - Reduce batch size in config
   - Decrease image size
   - Enable gradient checkpointing

3. **Data Loading Errors**
   - Check file paths in config
   - Verify dataset directory structure
   - Ensure image/mask filename matching

4. **Instance Extraction Issues**
   - Adjust `min_instance_size` parameter
   - Try different algorithms (watershed vs connected_components)
   - Check semantic mask quality

### Environment Testing
```bash
# Test installation
python -c "import unet; print('U-Net package ready!')"

# Test specific components
python -c "from unet import UNet; print('U-Net model ready!')"
python -c "from unet import InstanceExtractor; print('Instance extraction ready!')"
```

## 🛠️ Development

### Adding New Models
1. Create model file in `src/unet/models/`
2. Implement `forward()` method
3. Add to `src/unet/models/__init__.py`
4. Update trainer configuration

### Adding New Datasets
1. Create dataset class in `src/unet/data/`
2. Implement required methods
3. Add to `src/unet/data/__init__.py`
4. Update configuration options

### Adding New Instance Algorithms
1. Create algorithm file in `src/unet/instance_extraction/algorithms/`
2. Inherit from `BaseInstanceAlgorithm`
3. Implement required methods
4. Add to `src/unet/instance_extraction/__init__.py`

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run the test suite: `python -m pytest tests/`
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **U-Net Paper**: "U-Net: Convolutional Networks for Biomedical Image Segmentation"
- **BDD100K Dataset**: Berkeley DeepDrive 100K dataset
- **KITTI Dataset**: Karlsruhe Institute of Technology
- **PyTorch Team**: Excellent deep learning framework
- **Albumentations**: Comprehensive data augmentation library
- **segmentation-models-pytorch**: Pre-trained model implementations