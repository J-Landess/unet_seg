# U-Net Test Suite

This package contains comprehensive tests for the U-Net semantic segmentation project.

## 📁 Structure

```
tests/
├── __init__.py                 # Package initialization
├── test_video_inference.py     # Video inference testing
├── test_model_comparison.py    # Model comparison utilities
├── test_training.py           # Training functionality tests
├── run_tests.py               # Main test runner
└── README.md                  # This file
```

## 🚀 Quick Start

### Run All Tests
```bash
python tests/run_tests.py
```

### Run Specific Tests
```bash
# Video inference only
python tests/run_tests.py --tests inference

# Model comparison only  
python tests/run_tests.py --tests comparison

# Training tests only
python tests/run_tests.py --tests training

# Custom parameters
python tests/run_tests.py --max-frames 10 --epochs 5
```

## 🧪 Test Modules

### 1. Video Inference Tests (`test_video_inference.py`)

Tests video processing with different model types:

- **Dummy Segmentation**: Geometric patterns for pipeline testing
- **Trained Models**: Your custom trained models
- **Pre-trained Models**: VGG11, EfficientNet, etc.

```python
from tests.test_video_inference import VideoInferenceTester

tester = VideoInferenceTester()
tester.test_dummy_inference("video.mp4", max_frames=5)
tester.test_trained_model_inference("video.mp4", "model.pth")
tester.test_pretrained_inference("video.mp4", encoder_name="vgg11")
```

### 2. Model Comparison Tests (`test_model_comparison.py`)

Compare different models and analyze results:

- **Performance Metrics**: Speed, accuracy, file sizes
- **Quality Analysis**: Class distribution, segmentation quality
- **Visualization**: Side-by-side comparisons

```python
from tests.test_model_comparison import ModelComparisonTester

tester = ModelComparisonTester()
results = tester.compare_all_models()
quality = tester.analyze_segmentation_quality()
```

### 3. Training Tests (`test_training.py`)

Test training functionality:

- **Simple Training**: From-scratch training with dummy data
- **Pre-trained Training**: Transfer learning with pre-trained encoders
- **Performance Monitoring**: Loss tracking, validation metrics

```python
from tests.test_training import TrainingTester

tester = TrainingTester()
simple_results = tester.test_simple_training(num_epochs=5)
pretrained_results = tester.test_pretrained_training(encoder_name="vgg11")
```

## 📊 Output Structure

All test outputs are saved to `test_outputs/`:

```
test_outputs/
├── dummy_inference/           # Dummy segmentation results
├── trained_inference/         # Your trained model results
├── pretrained_inference/      # Pre-trained model results
├── training/                  # Training test results
└── comparison/                # Model comparison results
```

Each inference test generates:
- `*_original.jpg` - Original video frames
- `*_mask.png` - Segmentation masks
- `*_visualization.jpg` - Colored segmentations
- `*_overlay.jpg` - Original + segmentation overlays

## ⚙️ Configuration

### Test Parameters

- `--video`: Path to test video file
- `--max-frames`: Maximum frames to process (default: 5)
- `--epochs`: Training epochs for tests (default: 3)
- `--tests`: Which tests to run (inference, comparison, training, all)

### Model Configuration

Tests use the same configuration system as the main project:
- `config/config_cpu.yaml` - CPU-optimized settings
- `config/config.yaml` - Default settings

## 🔧 Customization

### Adding New Tests

1. Create a new test module in `tests/`
2. Inherit from base classes or create standalone functions
3. Add to `run_tests.py` if needed
4. Update this README

### Custom Model Testing

```python
# Add custom model to VideoInferenceTester
def test_custom_model(self, video_path, model_path):
    # Your custom model testing code
    pass
```

### Custom Metrics

```python
# Add custom metrics to ModelComparisonTester
def analyze_custom_metric(self):
    # Your custom analysis code
    pass
```

## 📈 Performance Benchmarks

The test suite provides performance benchmarks:

| Model Type | Parameters | Speed (FPS) | Memory | Best For |
|------------|------------|-------------|---------|----------|
| Dummy | N/A | ~0.5 | Low | Testing |
| Trained | 31M | ~0.2 | Medium | Custom |
| VGG11 | 18M | ~0.15 | Medium | General |
| EfficientNet | 6M | ~0.15 | Low | Efficient |

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **CUDA Issues**: Tests default to CPU, modify for GPU if needed
3. **Memory Issues**: Reduce `max_frames` or `batch_size`
4. **Model Not Found**: Check model paths and file existence

### Debug Mode

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

When adding new tests:

1. Follow the existing naming conventions
2. Add proper docstrings and type hints
3. Include error handling
4. Update this README
5. Test with different model types

## 📝 Examples

### Complete Test Workflow

```bash
# 1. Run all tests
python tests/run_tests.py --max-frames 10

# 2. Compare results
python tests/run_tests.py --tests comparison

# 3. Train and test new model
python tests/run_tests.py --tests training --epochs 10
python tests/run_tests.py --tests inference
```

### Programmatic Usage

```python
from tests import VideoInferenceTester, ModelComparisonTester

# Test video inference
tester = VideoInferenceTester()
results = tester.test_dummy_inference("my_video.mp4")

# Compare models
comparer = ModelComparisonTester()
comparison = comparer.compare_all_models()
```

This test suite provides a comprehensive way to validate and compare different approaches to semantic segmentation with your U-Net framework! 🎯
