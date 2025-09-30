# Instance Extraction Package

This package provides algorithms for extracting individual instances from semantic segmentation masks. It's designed to work seamlessly with your U-Net semantic segmentation pipeline and is extensible for future SOLO model integration.

## Features

- **Multiple Algorithms**: Watershed and Connected Components
- **Extensible Design**: Ready for SOLO model integration
- **Comprehensive Visualization**: Overlay, contours, labels, and comparison plots
- **Post-processing**: Size filtering, class filtering, instance merging
- **Metrics**: IoU-based evaluation and statistics
- **Easy Integration**: Simple API that works with existing U-Net outputs

## Quick Start

```python
from instance_extraction import InstanceExtractor

# Create extractor
extractor = InstanceExtractor(algorithm='watershed')

# Extract instances from semantic mask
instances = extractor.extract_instances(
    semantic_mask=semantic_mask,
    target_classes=[2, 11, 13],  # buildings, people, cars
    min_instance_size=100
)

# Visualize results
vis_image = extractor.visualize_instances(
    instances=instances,
    original_image=original_image,
    output_path="instances.jpg"
)
```

## Algorithms

### Watershed Algorithm
- **Best for**: Separating touching objects of the same class
- **Method**: Uses distance transform and local maxima as seeds
- **Parameters**: `min_distance`, `threshold_abs`, `min_separation`

### Connected Components Algorithm
- **Best for**: Simple, fast instance extraction
- **Method**: Uses connected component analysis
- **Parameters**: `connectivity`, `min_area`, `max_area`

## API Reference

### InstanceExtractor

Main interface for instance extraction.

```python
extractor = InstanceExtractor(algorithm='watershed', **kwargs)
```

**Methods:**
- `extract_instances(semantic_mask, target_classes, min_instance_size)`
- `visualize_instances(instances, original_image, output_path)`
- `get_instance_statistics(instances)`

### Post-processing

```python
from instance_extraction.utils import InstancePostProcessor

postprocessor = InstancePostProcessor()

# Filter by size
filtered = postprocessor.filter_instances_by_size(instances, min_size=100)

# Filter by class
filtered = postprocessor.filter_instances_by_class(instances, target_classes=[13])

# Merge small instances
merged = postprocessor.merge_small_instances(instances, max_size=50)
```

### Metrics

```python
from instance_extraction.utils import InstanceMetrics

metrics_calc = InstanceMetrics(iou_threshold=0.5)
metrics = metrics_calc.compute_metrics(pred_instances, gt_instances)
```

## Integration with U-Net

```python
# After U-Net inference
semantic_mask = model.predict(image)  # Your U-Net output

# Extract instances
extractor = InstanceExtractor(algorithm='watershed')
instances = extractor.extract_instances(
    semantic_mask=semantic_mask,
    target_classes=[2, 11, 13],  # BDD100K classes
    min_instance_size=100
)

# Visualize
vis_image = extractor.visualize_instances(
    instances=instances,
    original_image=image,
    show_overlay=True,
    show_contours=True,
    show_labels=True
)
```

## Future SOLO Integration

The package is designed to easily integrate with SOLO models:

```python
# Future SOLO integration (not yet implemented)
extractor = InstanceExtractor(algorithm='solo', model_path='solo_model.pth')
instances = extractor.extract_instances(semantic_mask)
```

## Examples

See `example_instance_integration.py` for a complete pipeline example.

## Dependencies

- numpy
- opencv-python
- scikit-image
- scipy
- matplotlib

## Installation

The package is part of your U-Net project. No additional installation required.

## Testing

Run the test suite:

```bash
python3 test_instance_extraction.py
```

## Output Files

The package generates:
- Instance visualizations with colored masks
- Overlay images showing instances on original images
- Comparison plots (semantic vs instance segmentation)
- Instance statistics and metrics
