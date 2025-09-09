#!/usr/bin/env python3
"""
Test script for instance extraction package
"""

import numpy as np
import cv2
import torch
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from instance_extraction import InstanceExtractor
from instance_extraction.algorithms import WatershedExtractor, ConnectedComponentsExtractor
from instance_extraction.visualization import InstanceVisualizer


def create_test_semantic_mask(size=(256, 256)):
    """Create a test semantic segmentation mask with multiple instances"""
    h, w = size
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Add some cars (class 13)
    for i in range(3):
        car_h = np.random.randint(20, 40)
        car_w = np.random.randint(30, 60)
        car_y = np.random.randint(h//2, h - car_h)
        car_x = np.random.randint(0, w - car_w)
        mask[car_y:car_y+car_h, car_x:car_x+car_w] = 13
    
    # Add some people (class 11)
    for i in range(2):
        person_h = np.random.randint(15, 25)
        person_w = np.random.randint(8, 15)
        person_y = np.random.randint(h//2, h - person_h)
        person_x = np.random.randint(0, w - person_w)
        mask[person_y:person_y+person_h, person_x:person_x+person_w] = 11
    
    # Add buildings (class 2)
    for i in range(2):
        building_h = np.random.randint(40, 80)
        building_w = np.random.randint(50, 100)
        building_y = np.random.randint(0, h//2)
        building_x = np.random.randint(0, w - building_w)
        mask[building_y:building_y+building_h, building_x:building_x+building_w] = 2
    
    # Add sky (class 10)
    mask[:h//3, :] = 10
    
    # Add road (class 0)
    mask[2*h//3:, :] = 0
    
    return mask


def create_test_image(size=(256, 256)):
    """Create a test RGB image"""
    h, w = size
    image = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    
    # Add some structure
    image[:h//3, :] = [135, 206, 235]  # Sky blue
    image[2*h//3:, :] = [105, 105, 105]  # Road gray
    
    return image


def test_watershed_extraction():
    """Test watershed-based instance extraction"""
    print("🌊 Testing Watershed Instance Extraction")
    print("=" * 50)
    
    # Create test data
    semantic_mask = create_test_semantic_mask()
    test_image = create_test_image()
    
    # Create extractor
    extractor = InstanceExtractor(algorithm='watershed')
    
    # Extract instances
    instances = extractor.extract_instances(
        semantic_mask=semantic_mask,
        target_classes=[2, 11, 13],  # buildings, people, cars
        min_instance_size=50
    )
    
    # Print results
    print(f"Found {len(instances['class_mapping'])} instances")
    print("Instance mapping:", instances['class_mapping'])
    print("Instance counts:", instances['instance_count'])
    
    # Visualize results
    output_dir = Path("test_outputs/instance_extraction")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    vis_image = extractor.visualize_instances(
        instances=instances,
        original_image=test_image,
        output_path=output_dir / "watershed_instances.jpg",
        show_overlay=True,
        show_contours=True,
        show_labels=True
    )
    
    print(f"✅ Watershed visualization saved to {output_dir / 'watershed_instances.jpg'}")
    
    return instances


def test_connected_components_extraction():
    """Test connected components-based instance extraction"""
    print("\n🔗 Testing Connected Components Instance Extraction")
    print("=" * 50)
    
    # Create test data
    semantic_mask = create_test_semantic_mask()
    test_image = create_test_image()
    
    # Create extractor
    extractor = InstanceExtractor(algorithm='connected_components')
    
    # Extract instances
    instances = extractor.extract_instances(
        semantic_mask=semantic_mask,
        target_classes=[2, 11, 13],  # buildings, people, cars
        min_instance_size=50
    )
    
    # Print results
    print(f"Found {len(instances['class_mapping'])} instances")
    print("Instance mapping:", instances['class_mapping'])
    print("Instance counts:", instances['instance_count'])
    
    # Visualize results
    output_dir = Path("test_outputs/instance_extraction")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    vis_image = extractor.visualize_instances(
        instances=instances,
        original_image=test_image,
        output_path=output_dir / "connected_components_instances.jpg",
        show_overlay=True,
        show_contours=True,
        show_labels=True
    )
    
    print(f"✅ Connected Components visualization saved to {output_dir / 'connected_components_instances.jpg'}")
    
    return instances


def test_postprocessing():
    """Test instance post-processing"""
    print("\n🔧 Testing Instance Post-Processing")
    print("=" * 50)
    
    # Create test data
    semantic_mask = create_test_semantic_mask()
    
    # Extract instances
    extractor = InstanceExtractor(algorithm='watershed')
    instances = extractor.extract_instances(
        semantic_mask=semantic_mask,
        target_classes=[2, 11, 13],
        min_instance_size=20  # Lower threshold to get more instances
    )
    
    print(f"Original instances: {len(instances['class_mapping'])}")
    
    # Test post-processing
    from instance_extraction.utils import InstancePostProcessor
    
    postprocessor = InstancePostProcessor()
    
    # Filter by size
    filtered_instances = postprocessor.filter_instances_by_size(
        instances, min_size=100, max_size=2000
    )
    print(f"After size filtering: {len(filtered_instances['class_mapping'])}")
    
    # Filter by class
    class_filtered = postprocessor.filter_instances_by_class(
        instances, target_classes=[13]  # Only cars
    )
    print(f"After class filtering (cars only): {len(class_filtered['class_mapping'])}")
    
    # Merge small instances
    merged_instances = postprocessor.merge_small_instances(
        instances, max_size=80, merge_strategy='nearest'
    )
    print(f"After merging small instances: {len(merged_instances['class_mapping'])}")
    
    print("✅ Post-processing tests completed")


def test_metrics():
    """Test instance segmentation metrics"""
    print("\n📊 Testing Instance Segmentation Metrics")
    print("=" * 50)
    
    # Create test data
    semantic_mask = create_test_semantic_mask()
    
    # Extract instances with different algorithms
    watershed_extractor = InstanceExtractor(algorithm='watershed')
    cc_extractor = InstanceExtractor(algorithm='connected_components')
    
    watershed_instances = watershed_extractor.extract_instances(
        semantic_mask=semantic_mask,
        target_classes=[2, 11, 13],
        min_instance_size=50
    )
    
    cc_instances = cc_extractor.extract_instances(
        semantic_mask=semantic_mask,
        target_classes=[2, 11, 13],
        min_instance_size=50
    )
    
    # Compute metrics
    from instance_extraction.utils import InstanceMetrics
    
    metrics_calc = InstanceMetrics(iou_threshold=0.5)
    
    # Compare algorithms
    metrics = metrics_calc.compute_metrics(watershed_instances, cc_instances)
    print("Watershed vs Connected Components metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Compute statistics
    watershed_stats = metrics_calc.compute_instance_statistics(watershed_instances)
    cc_stats = metrics_calc.compute_instance_statistics(cc_instances)
    
    print(f"\nWatershed statistics:")
    print(f"  Instances: {watershed_stats['num_instances']}")
    print(f"  Avg size: {watershed_stats['avg_instance_size']:.1f}")
    
    print(f"\nConnected Components statistics:")
    print(f"  Instances: {cc_stats['num_instances']}")
    print(f"  Avg size: {cc_stats['avg_instance_size']:.1f}")
    
    print("✅ Metrics tests completed")


def test_comparison_plot():
    """Test comparison visualization"""
    print("\n📈 Testing Comparison Visualization")
    print("=" * 50)
    
    # Create test data
    semantic_mask = create_test_semantic_mask()
    test_image = create_test_image()
    
    # Extract instances
    extractor = InstanceExtractor(algorithm='watershed')
    instances = extractor.extract_instances(
        semantic_mask=semantic_mask,
        target_classes=[2, 11, 13],
        min_instance_size=50
    )
    
    # Create comparison plot
    output_dir = Path("test_outputs/instance_extraction")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    extractor.visualizer.create_comparison_plot(
        original_image=test_image,
        semantic_mask=semantic_mask,
        instances=instances,
        output_path=output_dir / "comparison_plot.png"
    )
    
    print(f"✅ Comparison plot saved to {output_dir / 'comparison_plot.png'}")


def main():
    """Run all tests"""
    print("🧪 Instance Extraction Package Tests")
    print("=" * 60)
    
    try:
        # Test watershed extraction
        watershed_instances = test_watershed_extraction()
        
        # Test connected components extraction
        cc_instances = test_connected_components_extraction()
        
        # Test post-processing
        test_postprocessing()
        
        # Test metrics
        test_metrics()
        
        # Test comparison plot
        test_comparison_plot()
        
        print("\n🎉 All tests completed successfully!")
        print("\nGenerated files:")
        print("  - test_outputs/instance_extraction/watershed_instances.jpg")
        print("  - test_outputs/instance_extraction/connected_components_instances.jpg")
        print("  - test_outputs/instance_extraction/comparison_plot.png")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
