# 🚀 Installation Guide for U-Net Semantic Segmentation

This guide provides multiple installation options for the U-Net semantic segmentation project with BDD100K and KITTI dataset support.

## 📋 **Prerequisites**

- Python 3.10+ (recommended)
- CUDA 11.8+ (optional, for GPU acceleration)
- 8GB+ RAM (16GB+ recommended for large datasets)
- 10GB+ free disk space (for datasets)

## 🎯 **Installation Options**

### **Option 1: Conda Environment (Recommended)**

```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate unet-semantic-segmentation

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "from data import list_available_datasets; list_available_datasets()"
```

### **Option 2: Pip Installation**

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
```

### **Option 3: System-wide Installation (Not Recommended)**

```bash
# Install system-wide (use with caution)
pip install --user -r requirements.txt
```

## 🔧 **Environment Configuration**

### **CUDA Support (Optional)**

For GPU acceleration, uncomment the CUDA line in `environment.yml`:

```yaml
- pytorch-cuda=11.8  # Uncomment for CUDA support
```

Or install with pip:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### **macOS OpenCV Fix**

If you encounter OpenCV issues on macOS:

```bash
# Uninstall problematic opencv-python
pip uninstall opencv-python

# Install via conda (recommended)
conda install opencv -c conda-forge

# Or install specific version
pip install opencv-python==4.8.1.78
```

## 🧪 **Verification Tests**

### **Basic Import Test**

```python
# Test core imports
import torch
import torchvision
import cv2
import numpy as np
import albumentations as A
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
import yaml
import tqdm
import wandb
import requests

print("✅ All core packages imported successfully!")
```

### **Dataset Module Test**

```python
# Test dataset functionality
from data import list_available_datasets, create_sample_dataset_for_testing

# Show available datasets
datasets = list_available_datasets()

# Create sample data for testing
bdd_path, kitti_path = create_sample_dataset_for_testing("test_data")
print(f"✅ Sample datasets created: {bdd_path}, {kitti_path}")
```

### **U-Net Model Test**

```python
# Test U-Net model
from models import UNet
import torch

# Create model
model = UNet(n_channels=3, n_classes=19)
print(f"✅ U-Net model created: {sum(p.numel() for p in model.parameters())} parameters")

# Test forward pass
dummy_input = torch.randn(1, 3, 256, 256)
with torch.no_grad():
    output = model(dummy_input)
print(f"✅ Forward pass successful: {dummy_input.shape} → {output.shape}")
```

### **Complete Integration Test**

```python
# Test complete pipeline
from data import BDD100KSegmentationDataset
from models import UNet
import torch

# Create sample dataset
bdd_path, _ = create_sample_dataset_for_testing("integration_test")
dataset = BDD100KSegmentationDataset(str(bdd_path), image_size=(128, 128))

if len(dataset) > 0:
    # Load sample
    image, mask = dataset[0]
    
    # Create model
    model = UNet(n_channels=3, n_classes=19)
    model.eval()
    
    # Run inference
    with torch.no_grad():
        input_batch = image.unsqueeze(0)
        output = model(input_batch)
        prediction = torch.argmax(output, dim=1)
    
    print(f"✅ Complete pipeline working:")
    print(f"   Input: {input_batch.shape}")
    print(f"   Output: {output.shape}")
    print(f"   Prediction: {prediction.shape}")
```

## 📦 **Package Details**

### **Core Dependencies**

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥2.0.0 | Deep learning framework |
| `torchvision` | ≥0.15.0 | Computer vision utilities |
| `opencv-python` | ≥4.8.0 | Image/video processing |
| `numpy` | ≥1.24.0 | Numerical computing |
| `Pillow` | ≥9.5.0 | Image I/O |
| `albumentations` | ≥1.3.0 | Data augmentation |

### **Visualization & Analysis**

| Package | Version | Purpose |
|---------|---------|---------|
| `matplotlib` | ≥3.7.0 | Plotting |
| `seaborn` | ≥0.12.0 | Statistical visualization |
| `scikit-learn` | ≥1.3.0 | ML metrics |

### **Experiment Tracking**

| Package | Version | Purpose |
|---------|---------|---------|
| `tensorboard` | ≥2.13.0 | Training visualization |
| `wandb` | ≥0.15.0 | Experiment tracking |
| `tqdm` | ≥4.65.0 | Progress bars |

### **Utilities**

| Package | Version | Purpose |
|---------|---------|---------|
| `pyyaml` | ≥6.0 | Configuration files |
| `requests` | ≥2.28.0 | Dataset downloading |

## 🐛 **Troubleshooting**

### **Common Issues**

1. **OpenCV Import Error (macOS)**
   ```bash
   conda install opencv -c conda-forge
   ```

2. **PyTorch CUDA Issues**
   ```bash
   # Check CUDA availability
   python -c "import torch; print(torch.cuda.is_available())"
   
   # Reinstall with correct CUDA version
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Memory Issues**
   ```bash
   # Reduce batch size in config
   # Use smaller image sizes for testing
   # Enable gradient checkpointing
   ```

4. **Dataset Download Issues**
   ```bash
   # Install additional dependencies
   pip install requests tqdm
   
   # Check internet connection
   # Verify dataset URLs are accessible
   ```

### **Environment Verification**

```bash
# Check Python version
python --version

# Check package versions
pip list | grep -E "(torch|opencv|numpy|albumentations)"

# Test GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test OpenCV
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
```

## 🚀 **Quick Start**

After installation, test everything works:

```bash
# 1. Activate environment
conda activate unet-semantic-segmentation

# 2. Run basic test
python -c "from data import list_available_datasets; list_available_datasets()"

# 3. Create sample data
python -c "from data import create_sample_dataset_for_testing; create_sample_dataset_for_testing()"

# 4. Test U-Net
python -c "from models import UNet; model = UNet(); print('U-Net ready!')"

# 5. Run main script
python main.py --help
```

## 📚 **Next Steps**

1. **Download Real Datasets**: Use `setup_bdd100k_directory()` or `setup_kitti_directory()`
2. **Configure Training**: Edit `config/config.yaml`
3. **Start Training**: `python main.py train --config config/config.yaml`
4. **Run Inference**: `python main.py infer --model path/to/model.pth --input path/to/image.jpg`

Your environment is now ready for semantic segmentation with BDD100K and KITTI datasets! 🎉
