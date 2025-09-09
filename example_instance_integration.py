#!/usr/bin/env python3
"""
Example integration of instance extraction with U-Net semantic segmentation
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
from instance_extraction.utils import InstancePostProcessor


def run_semantic_to_instance_pipeline(
    semantic_mask: np.ndarray,
    original_image: np.ndarray,
    algorithm: str = 'watershed',
    target_classes: list = None,
    min_instance_size: int = 100
):
    """
    Complete pipeline from semantic segmentation to instance segmentation
    
    Args:
        semantic_mask: Semantic segmentation mask from U-Net
        original_image: Original RGB image
        algorithm: Instance extraction algorithm ('watershed' or 'connected_components')
        target_classes: List of class IDs to extract instances for
        min_instance_size: Minimum size for valid instances
        
    Returns:
        Dictionary with instance segmentation results and visualizations
    """
    
    print(f"🔍 Running {algorithm} instance extraction pipeline")
    print("=" * 60)
    
    # Initialize instance extractor
    extractor = InstanceExtractor(algorithm=algorithm)
    
    # Extract instances
    instances = extractor.extract_instances(
        semantic_mask=semantic_mask,
        target_classes=target_classes,
        min_instance_size=min_instance_size
    )
    
    # Print results
    print(f"Found {len(instances['class_mapping'])} instances")
    print("Instance mapping:", instances['class_mapping'])
    print("Instance counts:", instances['instance_count'])
    
    # Post-process instances (optional)
    postprocessor = InstancePostProcessor()
    
    # Filter by size
    filtered_instances = postprocessor.filter_instances_by_size(
        instances, min_size=min_instance_size
    )
    
    print(f"After size filtering: {len(filtered_instances['class_mapping'])} instances")
    
    # Create visualizations
    output_dir = Path("test_outputs/instance_integration")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Original semantic visualization
    semantic_vis = create_semantic_visualization(semantic_mask)
    cv2.imwrite(str(output_dir / "semantic_segmentation.jpg"), 
                cv2.cvtColor(semantic_vis, cv2.COLOR_RGB2BGR))
    
    # Instance visualization
    instance_vis = extractor.visualize_instances(
        instances=filtered_instances,
        original_image=original_image,
        output_path=output_dir / f"{algorithm}_instances.jpg",
        show_overlay=True,
        show_contours=True,
        show_labels=True
    )
    
    # Comparison plot
    extractor.visualizer.create_comparison_plot(
        original_image=original_image,
        semantic_mask=semantic_mask,
        instances=filtered_instances,
        output_path=output_dir / f"{algorithm}_comparison.png"
    )
    
    # Compute statistics
    stats = extractor.get_instance_statistics(filtered_instances)
    print(f"\nInstance Statistics:")
    print(f"  Total instances: {stats['total_instances']}")
    print(f"  Average size: {stats.get('avg_instance_size', 0):.1f} pixels")
    print(f"  Size range: {stats.get('min_instance_size', 0)} - {stats.get('max_instance_size', 0)} pixels")
    
    print(f"\n✅ Results saved to {output_dir}/")
    
    return {
        'instances': filtered_instances,
        'statistics': stats,
        'visualizations': {
            'semantic': str(output_dir / "semantic_segmentation.jpg"),
            'instances': str(output_dir / f"{algorithm}_instances.jpg"),
            'comparison': str(output_dir / f"{algorithm}_comparison.png")
        }
    }


def create_semantic_visualization(semantic_mask: np.ndarray) -> np.ndarray:
    """Create colored visualization of semantic segmentation mask"""
    
    # BDD100K class colors
    class_colors = {
        0: (0, 0, 0),        # road - black
        1: (128, 128, 128),  # sidewalk - gray
        2: (70, 70, 70),     # building - dark gray
        3: (153, 153, 153),  # wall - light gray
        4: (190, 190, 190),  # fence - silver
        5: (220, 20, 60),    # pole - crimson
        6: (255, 0, 0),      # traffic_light - red
        7: (255, 255, 0),    # traffic_sign - yellow
        8: (0, 255, 0),      # vegetation - green
        9: (107, 142, 35),   # terrain - olive
        10: (135, 206, 235), # sky - sky blue
        11: (255, 20, 147),  # person - deep pink
        12: (255, 105, 180), # rider - hot pink
        13: (0, 0, 255),     # car - blue
        14: (255, 165, 0),   # truck - orange
        15: (255, 0, 255),   # bus - magenta
        16: (0, 255, 255),   # train - cyan
        17: (128, 0, 128),   # motorcycle - purple
        18: (255, 192, 203)  # bicycle - pink
    }
    
    h, w = semantic_mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    unique_classes = np.unique(semantic_mask)
    
    for class_id in unique_classes:
        if class_id == 0:  # Skip background
            continue
        
        color = class_colors.get(class_id, (128, 128, 128))
        mask = semantic_mask == class_id
        colored[mask] = color
    
    return colored


def create_test_data():
    """Create test data for demonstration"""
    
    # Create a more complex test image
    h, w = 256, 256
    image = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    
    # Add structure
    image[:h//3, :] = [135, 206, 235]  # Sky blue
    image[2*h//3:, :] = [105, 105, 105]  # Road gray
    
    # Create semantic mask with multiple instances
    semantic_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Sky
    semantic_mask[:h//3, :] = 10
    
    # Road
    semantic_mask[2*h//3:, :] = 0
    
    # Add multiple cars
    for i in range(4):
        car_h = np.random.randint(20, 35)
        car_w = np.random.randint(30, 50)
        car_y = np.random.randint(h//2, h - car_h)
        car_x = np.random.randint(0, w - car_w)
        semantic_mask[car_y:car_y+car_h, car_x:car_x+car_w] = 13
    
    # Add buildings
    for i in range(3):
        building_h = np.random.randint(40, 80)
        building_w = np.random.randint(50, 100)
        building_y = np.random.randint(0, h//2)
        building_x = np.random.randint(0, w - building_w)
        semantic_mask[building_y:building_y+building_h, building_x:building_x+building_w] = 2
    
    # Add people
    for i in range(2):
        person_h = np.random.randint(15, 25)
        person_w = np.random.randint(8, 15)
        person_y = np.random.randint(h//2, h - person_h)
        person_x = np.random.randint(0, w - person_w)
        semantic_mask[person_y:person_y+person_h, person_x:person_x+person_w] = 11
    
    return image, semantic_mask


def main():
    """Run the complete pipeline example"""
    
    print("🚗 U-Net + Instance Extraction Pipeline Example")
    print("=" * 60)
    
    # Create test data
    print("Creating test data...")
    original_image, semantic_mask = create_test_data()
    
    print(f"Semantic mask shape: {semantic_mask.shape}")
    print(f"Unique classes: {np.unique(semantic_mask)}")
    
    # Test both algorithms
    algorithms = ['watershed', 'connected_components']
    target_classes = [2, 11, 13]  # buildings, people, cars
    
    results = {}
    
    for algorithm in algorithms:
        print(f"\n{'='*60}")
        result = run_semantic_to_instance_pipeline(
            semantic_mask=semantic_mask,
            original_image=original_image,
            algorithm=algorithm,
            target_classes=target_classes,
            min_instance_size=50
        )
        results[algorithm] = result
    
    # Compare results
    print(f"\n{'='*60}")
    print("📊 Algorithm Comparison")
    print("=" * 60)
    
    for algorithm, result in results.items():
        stats = result['statistics']
        print(f"\n{algorithm.upper()}:")
        print(f"  Instances found: {stats['total_instances']}")
        print(f"  Average size: {stats.get('avg_instance_size', 0):.1f} pixels")
        print(f"  Classes detected: {list(stats['instances_per_class'].keys())}")
    
    print(f"\n🎉 Pipeline example completed!")
    print(f"Check test_outputs/instance_integration/ for visualizations")


if __name__ == "__main__":
    main()
