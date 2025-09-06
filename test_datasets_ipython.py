#!/usr/bin/env python3
"""
iPython-ready test script for BDD100K and KITTI datasets

Copy-paste sections into iPython to test the new dataset functionality.
"""

def test_dataset_imports():
    """Test 1: Import all dataset classes"""
    print("🧪 Test 1: Dataset Imports")
    
    try:
        from data import list_available_datasets, get_dataset_info
        print("✅ Data module imported")
        
        # List available datasets
        datasets = list_available_datasets()
        
        # Get info about each dataset
        for category, dataset_list in datasets.items():
            for dataset_name in dataset_list:
                get_dataset_info(dataset_name)
                print()
        
    except Exception as e:
        print(f"❌ Import error: {e}")


def test_sample_dataset_creation():
    """Test 2: Create sample datasets for testing"""
    print("🧪 Test 2: Sample Dataset Creation")
    
    try:
        from data import create_sample_dataset_for_testing
        
        # Create sample datasets (works without real data)
        bdd_path, kitti_path = create_sample_dataset_for_testing("ipython_test_datasets")
        
        print(f"✅ Sample datasets created:")
        print(f"  BDD100K: {bdd_path}")
        print(f"  KITTI: {kitti_path}")
        
        return bdd_path, kitti_path
        
    except Exception as e:
        print(f"❌ Sample creation error: {e}")
        return None, None


def test_bdd100k_dataset(bdd_path):
    """Test 3: BDD100K dataset functionality"""
    print("🧪 Test 3: BDD100K Dataset")
    
    if not bdd_path:
        print("❌ No BDD100K path provided")
        return
    
    try:
        from data import BDD100KSegmentationDataset
        
        # Create dataset
        dataset = BDD100KSegmentationDataset(
            root_dir=str(bdd_path),
            split='train',
            image_size=(256, 256)
        )
        
        print(f"✅ BDD100K dataset created: {len(dataset)} samples")
        
        if len(dataset) > 0:
            # Test loading a sample
            image, mask = dataset[0]
            print(f"✅ Sample loaded:")
            print(f"  Image: {image.shape if hasattr(image, 'shape') else type(image)}")
            print(f"  Mask: {mask.shape if hasattr(mask, 'shape') else type(mask)}")
            
            # Show class information
            print(f"✅ BDD100K Classes: {len(dataset.BDD100K_CLASSES)}")
            for class_id, class_name in list(dataset.BDD100K_CLASSES.items())[:5]:
                print(f"  {class_id}: {class_name}")
            print("  ...")
        
    except Exception as e:
        print(f"❌ BDD100K error: {e}")


def test_kitti_dataset(kitti_path):
    """Test 4: KITTI dataset functionality"""
    print("🧪 Test 4: KITTI Dataset")
    
    if not kitti_path:
        print("❌ No KITTI path provided")
        return
    
    try:
        from data import KITTISegmentationDataset, KITTISequenceDataset
        
        # Test single frame dataset
        dataset = KITTISegmentationDataset(
            root_dir=str(kitti_path),
            split='train',
            image_size=(384, 384)
        )
        
        print(f"✅ KITTI dataset created: {len(dataset)} samples")
        
        if len(dataset) > 0:
            image, mask = dataset[0]
            print(f"✅ Sample loaded:")
            print(f"  Image: {image.shape if hasattr(image, 'shape') else type(image)}")
            print(f"  Mask: {mask.shape if hasattr(mask, 'shape') else type(mask)}")
        
        # Test sequence dataset
        seq_dataset = KITTISequenceDataset(
            root_dir=str(kitti_path),
            sequence_length=3,
            image_size=(256, 256)
        )
        
        print(f"✅ KITTI sequence dataset: {len(seq_dataset)} sequences")
        
        if len(seq_dataset) > 0:
            images, masks = seq_dataset[0]
            print(f"✅ Sequence loaded: {len(images)} frames")
        
    except Exception as e:
        print(f"❌ KITTI error: {e}")


def test_dataset_analysis():
    """Test 5: Dataset analysis tools"""
    print("🧪 Test 5: Dataset Analysis")
    
    try:
        from data import DatasetPreprocessor, DatasetValidator, create_sample_dataset_for_testing
        
        # Create sample data
        bdd_path, kitti_path = create_sample_dataset_for_testing("analysis_test")
        
        if bdd_path:
            from data import BDD100KSegmentationDataset
            dataset = BDD100KSegmentationDataset(str(bdd_path))
            
            if len(dataset) > 0:
                # Analyze dataset
                print("📊 Analyzing dataset...")
                stats = DatasetPreprocessor.analyze_dataset(dataset, num_samples=3)
                print(f"✅ Analysis complete: {stats.get('total_samples', 0)} samples")
                
                # Validate dataset
                print("🔍 Validating dataset...")
                report = DatasetValidator.validate_dataset(dataset)
                DatasetValidator.print_validation_report(report)
        
    except Exception as e:
        print(f"❌ Analysis error: {e}")


def test_dataloader_creation():
    """Test 6: DataLoader creation"""
    print("🧪 Test 6: DataLoader Creation")
    
    try:
        from data import create_bdd100k_dataloaders, create_kitti_dataloaders, create_sample_dataset_for_testing
        
        # Create sample datasets
        bdd_path, kitti_path = create_sample_dataset_for_testing("dataloader_test")
        
        if bdd_path:
            print("Creating BDD100K dataloaders...")
            try:
                train_loader, val_loader = create_bdd100k_dataloaders(
                    root_dir=str(bdd_path),
                    batch_size=2,  # Small batch for testing
                    image_size=(128, 128)  # Small size for speed
                )
                print(f"✅ BDD100K dataloaders created")
                print(f"  Train batches: {len(train_loader)}")
                print(f"  Val batches: {len(val_loader)}")
                
                # Test iteration
                if len(train_loader) > 0:
                    batch_images, batch_masks = next(iter(train_loader))
                    print(f"✅ Batch loaded: images {batch_images.shape}, masks {batch_masks.shape}")
                
            except Exception as e:
                print(f"❌ BDD100K dataloader error: {e}")
        
        if kitti_path:
            print("\nCreating KITTI dataloaders...")
            try:
                train_loader, val_loader = create_kitti_dataloaders(
                    root_dir=str(kitti_path),
                    batch_size=2,
                    image_size=(128, 128)
                )
                print(f"✅ KITTI dataloaders created")
                
            except Exception as e:
                print(f"❌ KITTI dataloader error: {e}")
        
    except Exception as e:
        print(f"❌ DataLoader test error: {e}")


# === COPY-PASTE READY FOR iPYTHON ===

def run_all_tests():
    """Run all dataset tests - COPY THIS TO iPYTHON"""
    print("🚀 Complete Dataset Testing Suite")
    print("=" * 50)
    
    # Test 1: Imports
    test_dataset_imports()
    print()
    
    # Test 2: Create samples
    bdd_path, kitti_path = test_sample_dataset_creation()
    print()
    
    # Test 3: BDD100K
    test_bdd100k_dataset(bdd_path)
    print()
    
    # Test 4: KITTI  
    test_kitti_dataset(kitti_path)
    print()
    
    # Test 5: Analysis
    test_dataset_analysis()
    print()
    
    # Test 6: DataLoaders
    test_dataloader_creation()
    
    print("\n🎉 All dataset tests complete!")
    print("\nNext steps:")
    print("1. Download real BDD100K or KITTI data")
    print("2. Use setup_bdd100k_directory() or setup_kitti_directory()")
    print("3. Place data in created structure")
    print("4. Create dataloaders with create_bdd100k_dataloaders() or create_kitti_dataloaders()")


if __name__ == "__main__":
    run_all_tests()
