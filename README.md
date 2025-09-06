# U-Net Semantic Segmentation

A comprehensive, production-ready implementation of U-Net for semantic segmentation with support for BDD100K and KITTI datasets, video processing, and advanced deep learning features.

## 🚀 **Features**

- **Complete U-Net Architecture**: 31M parameters with skip connections and customizable depth
- **Real-World Datasets**: BDD100K (100K driving images) and KITTI support
- **Video Processing**: Frame-by-frame video analysis with tensor output
- **Advanced Data Augmentation**: Albumentations with driving-specific augmentations
- **Professional Metrics**: Pixel accuracy, mean IoU, Dice coefficient, per-class metrics
- **Comprehensive Visualization**: Training curves, prediction overlays, dataset analysis
- **Flexible Configuration**: YAML-based configuration system
- **Experiment Tracking**: TensorBoard and Weights & Biases integration
- **Production Ready**: Complete CLI interface, error handling, logging

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
python test_environment.py
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

#### **Datasets**
```python
from data import BDD100KSegmentationDataset, KITTISegmentationDataset

# BDD100K dataset
bdd_dataset = BDD100KSegmentationDataset(
    root_dir="datasets/bdd100k",
    split='train',
    image_size=(512, 512)
)

# KITTI dataset
kitti_dataset = KITTISegmentationDataset(
    root_dir="datasets/kitti", 
    split='train',
    image_size=(512, 512)
)
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

## 📁 **Project Structure**

```
unet/
├── main.py                      # Main entry point
├── config/
│   └── config.yaml              # Configuration file
├── data/
│   ├── __init__.py
│   ├── dataset.py               # Generic dataset classes
│   ├── bdd100k_dataset.py       # BDD100K dataset implementation
│   ├── kitti_dataset.py         # KITTI dataset implementation
│   └── dataset_utils.py         # Dataset utilities and analysis
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
├── video_processing/
│   ├── __init__.py
│   ├── frame_iterator.py        # Video frame processing
│   └── README.md                # Video processing documentation
├── examples/                    # Example scripts
├── requirements.txt             # Pip dependencies
├── environment.yml              # Conda environment (general)
├── environment-macos.yml        # Conda environment (macOS)
├── environment-cuda.yml         # Conda environment (CUDA)
└── test_environment.py          # Environment testing script
```

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

## 📈 **Performance & Monitoring**

### **Metrics**
- **Pixel Accuracy**: Overall pixel-level accuracy
- **Mean IoU**: Intersection over Union for each class
- **Dice Coefficient**: Overlap measure for segmentation
- **Per-Class Metrics**: Detailed analysis for each class

### **Visualization**
- **Training Curves**: Loss and metric plots
- **Prediction Overlays**: Side-by-side comparisons
- **Confusion Matrices**: Class-wise performance
- **Dataset Analysis**: Class distribution and statistics

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

### **Environment Testing**
```bash
# Run comprehensive test
python test_environment.py

# Test specific components
python -c "from data import list_available_datasets; list_available_datasets()"
python -c "from models import UNet; print('U-Net ready!')"
```

## 📚 **Documentation**

- **Installation Guide**: `INSTALLATION_GUIDE.md`
- **Video Processing**: `video_processing/README.md`
- **Configuration Reference**: `config/config.yaml`
- **API Documentation**: Inline docstrings and type hints

## 🤝 **Contributing**

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 **License**

This project is open source and available under the MIT License.

## 🙏 **Acknowledgments**

- **U-Net Paper**: "U-Net: Convolutional Networks for Biomedical Image Segmentation"
- **BDD100K Dataset**: Berkeley DeepDrive 100K dataset
- **KITTI Dataset**: Karlsruhe Institute of Technology
- **PyTorch Team**: Excellent deep learning framework
- **Albumentations**: Comprehensive data augmentation library