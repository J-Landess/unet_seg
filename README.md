# U-Net Semantic Segmentation

U-Net for semantic segmentation with BDD100K / KITTI-oriented tooling, video helpers, instance extraction, YAML configs, and a CLI.

## Features

- U-Net architecture with skip connections
- BDD100K and KITTI-oriented data paths
- Video frame iteration / batching
- Instance extraction (watershed / connected components)
- Albumentations augmentations, metrics, visualization
- YAML configuration and CLI (`unet-train`, `unet-infer`, `unet-evaluate`, `unet-setup`)
- Optional pretrained encoders via **segmentation-models-pytorch** (used by some root scripts; not always listed as a hard install dep — install manually if needed)

## Installation

### Pip (recommended)

```bash
git clone <repository-url>
cd unet
pip install -e .
# or with dev extras
pip install -e .[dev]
```

### Manual

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Conda env files `environment-macos.yml` / `environment-cuda.yml` are **not present** in this repo. Prefer pip / `requirements.txt` above.

For root scripts such as `train_bdd100k_efficientnet.py` that import SMP:

```bash
pip install segmentation-models-pytorch
```

## Quick start

### CLI

`unet-infer` and `unet-evaluate` require `--config`.

```bash
unet-setup

unet-train --config config/config.yaml

unet-infer \
  --model checkpoints/best_model.pth \
  --config config/config.yaml \
  --input image.jpg \
  --output outputs/

unet-evaluate \
  --model checkpoints/best_model.pth \
  --config config/config.yaml \
  --dataset data/test/ \
  --output results/
```

### Python API

```python
from unet import SegmentationTrainer

trainer = SegmentationTrainer("config/config.yaml")
trainer.train()
```

```python
from unet import SegmentationInference

inference = SegmentationInference(
    model_path="checkpoints/best_model.pth",
    config_path="config/config.yaml",
)
prediction, probabilities = inference.predict_single("image.jpg")
```

## BDD100K helpers (repo root)

| File | Role |
|------|------|
| `BDD100K_DOWNLOAD_GUIDE.md` | Download / layout notes |
| `download_real_bdd100k.py` / `download_sample_data.py` | Download helpers |
| `setup_bdd100k_dataset.py` | Dataset layout setup |
| `train_bdd100k_efficientnet.py` | EfficientNet U-Net training (needs SMP) |
| `test_bdd100k_setup.py` | Setup smoke checks |
| `inference_pretrained.py` | Pretrained inference helper |

See **[BDD100K_DOWNLOAD_GUIDE.md](BDD100K_DOWNLOAD_GUIDE.md)**.

## Testing

Tests live under the package, not a top-level `tests/`:

```bash
python -m pytest src/unet/tests/

python -m pytest src/unet/tests/core_tests.py
python -m pytest src/unet/tests/inference_tests.py
python -m pytest src/unet/tests/training_tests.py

python -m pytest src/unet/tests/ --cov=unet --cov-report=html
```

## Configuration

Edit `config/config.yaml` for model, data, and training settings.

## Project structure

```
unet/
├── src/unet/                 # Package (models, data, training, inference, tests, …)
├── config/                   # YAML configs
├── assets/                   # Example images
├── BDD100K_DOWNLOAD_GUIDE.md
├── train_bdd100k_efficientnet.py
├── setup_bdd100k_dataset.py
├── pyproject.toml
└── requirements.txt
```

## License

MIT License
