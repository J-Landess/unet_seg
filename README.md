# U-Net Semantic Segmentation

A comprehensive, production-ready implementation of U-Net for semantic segmentation with support for BDD100K and KITTI datasets, video processing, instance extraction, and advanced deep learning features.

## 🚀 **Features**

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

## 📦 **Quick Start**

### **1. Installation**

**Option A: Conda (Recommended)**
```bash
# Create environment
conda env create -f environment-macos.yml  # macOS
# or
conda env create -f environment-cuda.yml   # Linux/Windows with CUDA

# Activate environment
conda activate unet-semantic-segmentation
```

**Option B: Pip**
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### **2. Verify Installation**
```bash
# Test everything works
python tests/core_tests.py

# Or run quick validation
python tests/run_tests.py --tests quick
```

### **3. Quick Test**
```python
# Copy-paste into iPython
from data import list_available_datasets, create_sample_dataset_for_testing
from models import UNet
import torch

# Show available datasets
list_available_datasets()

# Create sample data
bdd_path, kitti_path = create_sample_dataset_for_testing("test")

# Test U-Net
model = UNet(n_channels=3, n_classes=19)
print(f"✅ U-Net ready: {sum(p.numel() for p in model.parameters())} parameters")
```

## 🎯 **Usage**

### **Command Line Interface**

```bash
# Setup data structure
python main.py setup

# Train model
python main.py train --config config/config.yaml

# Run inference
python main.py infer --model checkpoints/best_model.pth --input image.jpg --output outputs/

# Evaluate model
python main.py evaluate --model checkpoints/best_model.pth --dataset data/test/ --output results/
```

### **Python API**

#### **Training**
```python
from training import SegmentationTrainer

# Initialize trainer
trainer = SegmentationTrainer("config/config.yaml")

# Start training
trainer.train()

# Resume from checkpoint
trainer.train(resume_from="checkpoints/checkpoint_epoch_50.pth")
```

#### **Inference**
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
```

#### **Video Processing**
```python
from video_processing import VideoFrameIterator, TensorFrameBatcher

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

#### **Instance Extraction**
```python
from instance_extraction import InstanceExtractor

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

## 📁 **Project Structure**

```
unet/
├── main.py                          # Main entry point
├── config/
│   ├── config.yaml                  # Main configuration
│   ├── config_cpu.yaml             # CPU-optimized config
│   └── config_cuda.yaml            # CUDA-optimized config
├── data/
│   ├── __init__.py
│   ├── dataset.py                   # Generic dataset classes
│   ├── bdd100k_dataset.py          # BDD100K dataset implementation
│   ├── kitti_dataset.py            # KITTI dataset implementation
│   └── dataset_utils.py            # Dataset utilities and analysis
├── models/
│   ├── __init__.py
│   └── unet.py                     # U-Net model implementation
├── training/
│   ├── __init__.py
│   └── trainer.py                  # Training pipeline
├── inference/
│   ├── __init__.py
│   └── inference.py                # Inference engine
├── instance_extraction/            # Instance segmentation subpackage
│   ├── __init__.py
│   ├── core.py                     # Main instance extractor
│   ├── base.py                     # Base classes
│   ├── algorithms/
│   │   ├── watershed.py            # Watershed algorithm
│   │   └── connected_components.py # Connected components algorithm
│   ├── utils/
│   │   ├── metrics.py              # Instance metrics
│   │   └── postprocessing.py       # Post-processing utilities
│   └── visualization/
│       └── visualizer.py           # Visualization tools
├── video_processing/
│   ├── __init__.py
│   └── frame_iterator.py           # Video frame processing
├── utils/
│   ├── __init__.py
│   ├── metrics.py                  # Evaluation metrics
│   └── visualization.py            # Visualization utilities
├── tests/                          # Consolidated test suite
│   ├── __init__.py
│   ├── run_tests.py                # Main test runner
│   ├── core_tests.py               # Core functionality tests
│   ├── inference_tests.py          # Inference testing
│   ├── training_tests.py           # Training and model tests
│   ├── test_model_comparison.py    # Model comparison tests
│   ├── test_training.py            # Legacy training tests
│   └── test_video_inference.py     # Legacy video inference tests
├── examples/                       # Example scripts and demos
│   ├── __init__.py
│   ├── demo_scripts.py             # Consolidated demo scripts
│   └── ipython_testing_guide.md    # iPython testing guide
├── unet_test/                      # Sample datasets
│   ├── bdd100k_sample/             # BDD100K sample data
│   └── kitti_sample/               # KITTI sample data
├── requirements.txt                # Pip dependencies
├── environment.yml                 # Conda environment (general)
├── environment-macos.yml           # Conda environment (macOS)
├── environment-cuda.yml            # Conda environment (CUDA)
├── train_bdd100k.py               # BDD100K training script
├── train_bdd100k_simple.py        # Simplified BDD100K training
├── pretrained_model_inference.py  # Pre-trained model inference
├── compare_all_models.py          # Model comparison utility
├── compare_inference_results.py   # Inference comparison utility
└── retrain_colorful.py            # Colorful retraining script
```

## 🧪 **Testing**

The project includes a comprehensive test suite organized by functionality:

### **Run All Tests**
```bash
# Run complete test suite
python tests/run_tests.py

# Run specific test categories
python tests/run_tests.py --tests core inference training demos

# Quick validation
python tests/run_tests.py --tests quick
```

### **Individual Test Modules**
```bash
# Core functionality tests
python tests/core_tests.py

# Inference tests
python tests/inference_tests.py

# Training tests
python tests/training_tests.py

# Demo scripts
python examples/demo_scripts.py
```

### **Test Categories**

- **Core Tests** (`tests/core_tests.py`): Environment setup, dataset functionality, video processing
- **Inference Tests** (`tests/inference_tests.py`): Dummy, trained, and pre-trained model inference
- **Training Tests** (`tests/training_tests.py`): Training pipeline, model architecture, BDD100K training
- **Demo Scripts** (`examples/demo_scripts.py`): Complete workflow demonstrations

## 🔧 **Configuration**

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

## 📊 **Supported Datasets**

### **BDD100K Dataset**
- **100K driving images** with 19 semantic classes
- **Weather augmentations**: fog, rain, shadow
- **Class balancing** with computed weights
- **Download**: https://bdd-data.berkeley.edu/

### **KITTI Dataset**
- **Autonomous driving scenes** with detailed labels
- **Sequence support** for video processing
- **Temporal analysis** capabilities
- **Download**: http://www.cvlibs.net/datasets/kitti/

### **Custom Datasets**
```python
from data import SegmentationDataset

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

## 🎥 **Video Processing**

The video processing module provides frame-by-frame analysis with tensor output:

```python
from video_processing import VideoFrameIterator, TensorFrameBatcher

# Basic video iteration
with VideoFrameIterator("video.mp4") as iterator:
    for frame, metadata in iterator:
        print(f"Frame {metadata.frame_number}: {frame.shape}")

# Tensor output for deep learning
with VideoFrameIterator("video.mp4", output_format="tensor") as iterator:
    for frame, metadata in iterator:
        # frame is a PyTorch tensor (C, H, W) normalized to [0,1]
        prediction = model(frame.unsqueeze(0))

# Batch processing
batcher = TensorFrameBatcher(batch_size=8)
# ... add frames
batch = batcher.get_batch()  # (B, C, H, W) tensor
```

## 🔍 **Instance Extraction**

Convert semantic segmentation masks to instance segmentation:

```python
from instance_extraction import InstanceExtractor

# Initialize with Watershed algorithm
extractor = InstanceExtractor(algorithm="watershed")

# Extract instances
instances = extractor.extract_instances(
    semantic_mask=semantic_mask,
    target_classes=[2, 11, 13],  # buildings, people, cars
    min_instance_size=50
)

# Visualize results
visualization = extractor.visualize_instances(
    instances=instances,
    original_image=image,
    output_path="output.jpg",
    show_overlay=True,
    show_contours=True,
    show_labels=True
)
```

## 🏗️ **Pre-trained Models**

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

## 📈 **Performance & Monitoring**

### **Metrics**
- **Pixel Accuracy**: Overall pixel-level accuracy
- **Mean IoU**: Intersection over Union for each class
- **Dice Coefficient**: Overlap measure for segmentation
- **Per-Class Metrics**: Detailed analysis for each class
- **Instance Metrics**: Instance-level evaluation

### **Visualization**
- **Training Curves**: Loss and metric plots
- **Prediction Overlays**: Side-by-side comparisons
- **Confusion Matrices**: Class-wise performance
- **Dataset Analysis**: Class distribution and statistics
- **Instance Visualizations**: Contour and label overlays

### **Experiment Tracking**
- **TensorBoard**: Real-time training monitoring
- **Weights & Biases**: Advanced experiment tracking
- **Logging**: Comprehensive training logs

## 🛠️ **Development**

### **Adding New Models**
1. Create model file in `models/`
2. Implement `forward()` method
3. Add to `models/__init__.py`
4. Update trainer configuration

### **Adding New Datasets**
1. Create dataset class in `data/`
2. Implement required methods
3. Add to `data/__init__.py`
4. Update configuration options

### **Adding New Instance Algorithms**
1. Create algorithm file in `instance_extraction/algorithms/`
2. Inherit from `BaseInstanceAlgorithm`
3. Implement required methods
4. Add to `instance_extraction/__init__.py`

### **Adding New Metrics**
1. Add metric functions to `utils/metrics.py`
2. Update `SegmentationMetrics` class
3. Add to `utils/__init__.py`

## 🔍 **Troubleshooting**

### **Common Issues**

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

### **Environment Testing**
```bash
# Run comprehensive test
python tests/core_tests.py

# Test specific components
python -c "from data import list_available_datasets; list_available_datasets()"
python -c "from models import UNet; print('U-Net ready!')"
python -c "from instance_extraction import InstanceExtractor; print('Instance extraction ready!')"
```

## 📚 **Documentation**

- **Installation Guide**: `INSTALLATION_GUIDE.md`
- **Video Processing**: `video_processing/README.md`
- **Configuration Reference**: `config/config.yaml`
- **API Documentation**: Inline docstrings and type hints
- **Test Documentation**: `tests/README.md`

## 🎭 **Examples & Demos**

The `examples/demo_scripts.py` module provides comprehensive demonstrations:

- **Instance Integration Demo**: Complete semantic to instance pipeline
- **Tensor Video Processing**: Deep learning preprocessing with batching
- **U-Net Video Segmentation**: Video segmentation with batch processing
- **Inference Examples**: Basic inference functionality
- **Training Examples**: Training pipeline demonstrations

Run demos:
```bash
python examples/demo_scripts.py
```

## 🤝 **Contributing**

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run the test suite: `python tests/run_tests.py`
5. Submit a pull request

## 📄 **License**

This project is open source and available under the MIT License.

## 🙏 **Acknowledgments**

- **U-Net Paper**: "U-Net: Convolutional Networks for Biomedical Image Segmentation"
- **BDD100K Dataset**: Berkeley DeepDrive 100K dataset
- **KITTI Dataset**: Karlsruhe Institute of Technology
- **PyTorch Team**: Excellent deep learning framework
- **Albumentations**: Comprehensive data augmentation library
- **segmentation-models-pytorch**: Pre-trained model implementations