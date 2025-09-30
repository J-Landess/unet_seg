# Pre-trained Models Integration Guide

This guide shows you how to integrate pre-trained U-Net models into your semantic segmentation framework.

## 🎯 Available Pre-trained Models

### 1. **TernausNet** (Recommended for General Use)
- **Architecture**: U-Net with VGG11 encoder pre-trained on ImageNet
- **Best for**: General semantic segmentation, transfer learning
- **Advantages**: Proven performance, easy to integrate
- **Paper**: [TernausNet: U-Net with VGG11 Encoder Pre-trained on ImageNet for Image Segmentation](https://arxiv.org/abs/1801.05746)

### 2. **NVIDIA Pre-trained Models**
- **Architecture**: U-Net with ResNet/VGG backbones
- **Best for**: Industrial applications, high performance
- **Advantages**: Optimized for NVIDIA hardware
- **Source**: [NVIDIA TLT Documentation](https://docs.nvidia.com/metropolis/TLT/)

### 3. **Hugging Face Models**
- **Architecture**: Various U-Net variants
- **Best for**: Quick experimentation, research
- **Advantages**: Easy to download and use
- **Source**: [Hugging Face Hub](https://huggingface.co/models?pipeline_tag=image-segmentation)

### 4. **Medical Image Models**
- **STU-Net**: Up to 1.4B parameters, pre-trained on TotalSegmentator
- **Swin-Unet**: Transformer-based U-Net
- **Best for**: Medical imaging applications

## 🚀 Integration Methods

### Method 1: Use Pre-trained Encoder Only (Transfer Learning)

```python
import torch
import torch.nn as nn
from torchvision import models

class PreTrainedUNet(nn.Module):
    def __init__(self, num_classes=21, pretrained=True):
        super(PreTrainedUNet, self).__init__()
        
        # Use pre-trained VGG11 as encoder
        vgg = models.vgg11_bn(pretrained=pretrained)
        self.encoder = vgg.features
        
        # Custom decoder
        self.decoder = self._make_decoder(num_classes)
    
    def _make_decoder(self, num_classes):
        return nn.Sequential(
            nn.ConvTranspose2d(512, 256, 2, 2),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.ConvTranspose2d(128, 64, 2, 2),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ConvTranspose2d(32, 16, 2, 2),
            nn.Conv2d(16, num_classes, 1)
        )
    
    def forward(self, x):
        # Encoder
        features = self.encoder(x)
        
        # Decoder
        out = self.decoder(features)
        
        return out
```

### Method 2: Load Pre-trained Weights

```python
def load_pretrained_weights(model, pretrained_path):
    """Load pre-trained weights into your model"""
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Load weights (skip mismatched layers)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() 
                      if k in model_dict and v.shape == model_dict[k].shape}
    
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    return model
```

### Method 3: Use Hugging Face Models

```python
from transformers import AutoModel, AutoImageProcessor

def load_huggingface_model(model_name="microsoft/unet-base"):
    """Load a pre-trained model from Hugging Face"""
    model = AutoModel.from_pretrained(model_name)
    processor = AutoImageProcessor.from_pretrained(model_name)
    return model, processor
```

## 🔧 Implementation Examples

### Example 1: TernausNet Integration

```python
# Add this to your models/__init__.py
def create_ternausnet_model(num_classes=21, pretrained=True):
    """Create TernausNet model with pre-trained VGG11 encoder"""
    from .ternausnet import TernausNet
    
    model = TernausNet(
        num_classes=num_classes,
        num_filters=32,
        is_deconv=False,
        pretrained=pretrained
    )
    
    return model
```

### Example 2: Update Your Training Script

```python
# Modify simple_training_test.py
def create_pretrained_model(num_classes=21, pretrained=True):
    """Create model with pre-trained encoder"""
    if pretrained:
        print("🔄 Loading pre-trained TernausNet...")
        model = create_ternausnet_model(num_classes, pretrained=True)
    else:
        print("🔄 Creating model from scratch...")
        model = SimpleUNet(num_classes=num_classes)
    
    return model
```

## 📦 Installation Requirements

```bash
# For TernausNet
pip install ternausnet

# For Hugging Face models
pip install transformers

# For additional pre-trained models
pip install timm  # PyTorch Image Models
pip install segmentation-models-pytorch
```

## 🎯 Recommended Approach for Your Project

### Step 1: Use Pre-trained Encoder
```python
# Update your config/config_cpu.yaml
model:
  name: "ternausnet"  # or "pretrained_unet"
  pretrained: true
  num_classes: 21
```

### Step 2: Fine-tune on Your Data
```python
# Train with pre-trained weights
python3 simple_training_test.py --pretrained
```

### Step 3: Compare Performance
```python
# Test both models
python3 test_trained_model_inference.py --model checkpoints/pretrained_model.pth
python3 test_trained_model_inference.py --model checkpoints/scratch_model.pth
```

## 🔍 Model Comparison

| Model Type | Training Time | Accuracy | Memory Usage | Best For |
|------------|---------------|----------|--------------|----------|
| **From Scratch** | Long | Good | High | Custom datasets |
| **Pre-trained Encoder** | Medium | Better | Medium | Transfer learning |
| **Fully Pre-trained** | Short | Best | Low | Quick deployment |

## 🚀 Quick Start

1. **Install dependencies**:
   ```bash
   pip install ternausnet segmentation-models-pytorch
   ```

2. **Use pre-trained model**:
   ```python
   from segmentation_models_pytorch import Unet
   
   model = Unet(
       encoder_name="vgg11",
       encoder_weights="imagenet",
       classes=21,
       activation=None
   )
   ```

3. **Fine-tune on your data**:
   ```python
   # Your existing training code will work with minimal changes
   ```

## 💡 Pro Tips

- **Start with pre-trained encoders** for better performance
- **Freeze early layers** during initial training
- **Use learning rate scheduling** for fine-tuning
- **Monitor validation loss** to prevent overfitting
- **Experiment with different backbones** (ResNet, EfficientNet, etc.)

## 🔗 Useful Resources

- [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch)
- [TernausNet GitHub](https://github.com/ternaus/TernausNet)
- [Hugging Face Image Segmentation](https://huggingface.co/models?pipeline_tag=image-segmentation)
- [PyTorch Hub](https://pytorch.org/hub/)
