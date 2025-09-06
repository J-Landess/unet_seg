#!/usr/bin/env python3
"""
Comprehensive Environment Test for U-Net Semantic Segmentation

This script tests all components of the U-Net project including:
- Package imports and versions
- Dataset functionality (BDD100K, KITTI)
- U-Net model integration
- Video processing capabilities
- Complete end-to-end pipeline
"""

import sys
import traceback
from pathlib import Path

def test_package_imports():
    """Test 1: Package Imports and Versions"""
    print("🧪 Test 1: Package Imports and Versions")
    print("-" * 50)
    
    try:
        import torch
        import torchvision
        import cv2
        import numpy as np
        import albumentations as A
        from PIL import Image
        import matplotlib.pyplot as plt
        import seaborn as sns
        import sklearn
        import tqdm
        import requests
        import yaml
        
        print("✅ All packages imported successfully!")
        print(f"  PyTorch: {torch.__version__}")
        print(f"  OpenCV: {cv2.__version__}")
        print(f"  NumPy: {np.__version__}")
        print(f"  Albumentations: {A.__version__}")
        print(f"  PIL: {Image.__version__}")
        print(f"  Matplotlib: {plt.matplotlib.__version__}")
        print(f"  Seaborn: {sns.__version__}")
        print(f"  Scikit-learn: {sklearn.__version__}")
        print(f"  YAML: {yaml.__version__}")
        print(f"  TQDM: {tqdm.__version__}")
        print(f"  Requests: {requests.__version__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Package import error: {e}")
        traceback.print_exc()
        return False

def test_dataset_functionality():
    """Test 2: Dataset Functionality"""
    print("\n🧪 Test 2: Dataset Functionality")
    print("-" * 50)
    
    try:
        from data import list_available_datasets, create_sample_dataset_for_testing
        from data import BDD100KSegmentationDataset, KITTISegmentationDataset
        
        # Test dataset listing
        print("📊 Available datasets:")
        datasets = list_available_datasets()
        
        # Test sample dataset creation
        print("\n🏗️  Creating sample datasets...")
        bdd_path, kitti_path = create_sample_dataset_for_testing("env_test")
        print(f"✅ Sample datasets created:")
        print(f"  BDD100K: {bdd_path}")
        print(f"  KITTI: {kitti_path}")
        
        # Test BDD100K dataset
        print("\n🚗 Testing BDD100K dataset...")
        bdd_dataset = BDD100KSegmentationDataset(str(bdd_path), image_size=(128, 128))
        print(f"✅ BDD100K: {len(bdd_dataset)} samples, {len(bdd_dataset.BDD100K_CLASSES)} classes")
        
        if len(bdd_dataset) > 0:
            image, mask = bdd_dataset[0]
            print(f"  Sample: {image.shape} {image.dtype}, {mask.shape} {mask.dtype}")
        
        # Test KITTI dataset
        print("\n🚗 Testing KITTI dataset...")
        kitti_dataset = KITTISegmentationDataset(str(kitti_path), image_size=(128, 128))
        print(f"✅ KITTI: {len(kitti_dataset)} samples, {len(kitti_dataset.KITTI_CLASSES)} classes")
        
        if len(kitti_dataset) > 0:
            image, mask = kitti_dataset[0]
            print(f"  Sample: {image.shape} {image.dtype}, {mask.shape} {mask.dtype}")
        
        return True, bdd_path, kitti_path
        
    except Exception as e:
        print(f"❌ Dataset functionality error: {e}")
        traceback.print_exc()
        return False, None, None

def test_unet_model():
    """Test 3: U-Net Model"""
    print("\n🧪 Test 3: U-Net Model")
    print("-" * 50)
    
    try:
        from models import UNet
        import torch
        
        # Create model
        model = UNet(n_channels=3, n_classes=19)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✅ U-Net model created:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✅ Forward pass successful:")
        print(f"  Input: {dummy_input.shape}")
        print(f"  Output: {output.shape}")
        
        return True, model
        
    except Exception as e:
        print(f"❌ U-Net model error: {e}")
        traceback.print_exc()
        return False, None

def test_end_to_end_pipeline(bdd_path, model):
    """Test 4: End-to-End Pipeline"""
    print("\n🧪 Test 4: End-to-End Pipeline")
    print("-" * 50)
    
    try:
        from data import BDD100KSegmentationDataset
        import torch
        
        if not bdd_path or not model:
            print("❌ Missing prerequisites for end-to-end test")
            return False
        
        # Create dataset
        dataset = BDD100KSegmentationDataset(str(bdd_path), image_size=(128, 128))
        
        if len(dataset) == 0:
            print("❌ No samples in dataset")
            return False
        
        # Load sample
        image, mask = dataset[0]
        print(f"✅ Sample loaded: {image.shape} {image.dtype}, {mask.shape} {mask.dtype}")
        
        # Ensure proper data types
        if image.dtype != torch.float32:
            image = image.float()
        if mask.dtype != torch.long:
            mask = mask.long()
        
        # Run inference
        model.eval()
        input_batch = image.unsqueeze(0)
        
        with torch.no_grad():
            output = model(input_batch)
            prediction = torch.argmax(output, dim=1)
        
        print(f"✅ End-to-end pipeline successful:")
        print(f"  Input: {input_batch.shape}")
        print(f"  Output: {output.shape}")
        print(f"  Prediction: {prediction.shape}")
        print(f"  Ground truth classes: {torch.unique(mask).tolist()}")
        print(f"  Predicted classes: {torch.unique(prediction).tolist()}")
        
        return True
        
    except Exception as e:
        print(f"❌ End-to-end pipeline error: {e}")
        traceback.print_exc()
        return False

def test_video_processing():
    """Test 5: Video Processing"""
    print("\n🧪 Test 5: Video Processing")
    print("-" * 50)
    
    try:
        from video_processing import VideoFrameIterator, TensorFrameBatcher
        import torch
        import numpy as np
        
        print("✅ Video processing modules imported")
        
        # Test tensor frame batcher
        batcher = TensorFrameBatcher(batch_size=2)
        print(f"✅ TensorFrameBatcher created: batch_size={batcher.batch_size}")
        
        # Test with dummy frames
        dummy_frames = [torch.randn(3, 64, 64) for _ in range(3)]
        
        for i, frame in enumerate(dummy_frames):
            # Create dummy metadata
            from video_processing import VideoFrameMetadata
            metadata = VideoFrameMetadata(
                frame_number=i,
                timestamp=float(i),
                video_info={'fps': 30, 'width': 64, 'height': 64}
            )
            batcher.add_frame(frame, metadata)
            print(f"  Added frame {i+1}: {frame.shape}")
        
        # Get batch
        batch = batcher.get_batch()
        if batch is not None:
            print(f"✅ Batch created: {batch.shape}")
        else:
            print("⚠️  No batch available (need more frames)")
        
        return True
        
    except Exception as e:
        print(f"❌ Video processing error: {e}")
        traceback.print_exc()
        return False

def test_dataloader_creation(bdd_path):
    """Test 6: DataLoader Creation"""
    print("\n🧪 Test 6: DataLoader Creation")
    print("-" * 50)
    
    try:
        from data import create_bdd100k_dataloaders
        import torch
        
        if not bdd_path:
            print("❌ No BDD100K path for dataloader test")
            return False
        
        # Create dataloaders
        train_loader, val_loader = create_bdd100k_dataloaders(
            root_dir=str(bdd_path),
            batch_size=2,
            image_size=(64, 64),
            num_workers=0  # Use 0 for testing
        )
        
        print(f"✅ DataLoaders created:")
        print(f"  Train: {len(train_loader)} batches")
        print(f"  Val: {len(val_loader)} batches")
        
        # Test batch loading
        if len(train_loader) > 0:
            batch_images, batch_masks = next(iter(train_loader))
            print(f"✅ Batch loaded successfully:")
            print(f"  Images: {batch_images.shape}")
            print(f"  Masks: {batch_masks.shape}")
            print(f"  Image range: [{batch_images.min():.3f}, {batch_images.max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ DataLoader creation error: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 U-Net Semantic Segmentation Environment Test")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Package imports
    results['packages'] = test_package_imports()
    
    # Test 2: Dataset functionality
    results['datasets'], bdd_path, kitti_path = test_dataset_functionality()
    
    # Test 3: U-Net model
    results['unet'], model = test_unet_model()
    
    # Test 4: End-to-end pipeline
    results['pipeline'] = test_end_to_end_pipeline(bdd_path, model)
    
    # Test 5: Video processing
    results['video'] = test_video_processing()
    
    # Test 6: DataLoader creation
    results['dataloader'] = test_dataloader_creation(bdd_path)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"  {test_name.upper()}: {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Your environment is ready for U-Net semantic segmentation!")
        print("\n🚀 Next steps:")
        print("  1. Download real BDD100K or KITTI datasets")
        print("  2. Use setup_bdd100k_directory() or setup_kitti_directory()")
        print("  3. Start training with: python main.py train --config config/config.yaml")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        print("💡 Try running: pip install -r requirements.txt")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
