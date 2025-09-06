# 📊 Dataset Testing in iPython

## 🚀 **Copy-Paste Ready Code for iPython**

### **Test 1: Basic Dataset Module (Copy This First)**

```python
# === Dataset Module Testing ===
print("🧪 Testing Enhanced Data Module with BDD100K & KITTI")

# Test imports
try:
    from data import list_available_datasets, get_dataset_info
    print("✅ Data module imported successfully")
    
    # Show available datasets
    datasets = list_available_datasets()
    
except Exception as e:
    print(f"❌ Import error: {e}")
```

### **Test 2: Create Sample Datasets (Copy This Next)**

```python
# === Create Sample Datasets for Testing ===
print("\n🏗️  Creating Sample Datasets")

try:
    from data import create_sample_dataset_for_testing
    
    # Create sample BDD100K and KITTI datasets
    bdd_path, kitti_path = create_sample_dataset_for_testing("ipython_datasets")
    
    print(f"✅ Sample datasets created:")
    print(f"  BDD100K: {bdd_path}")
    print(f"  KITTI: {kitti_path}")
    
    # Store paths for later tests
    BDD_PATH = str(bdd_path)
    KITTI_PATH = str(kitti_path)
    
except Exception as e:
    print(f"❌ Sample creation error: {e}")
    BDD_PATH = None
    KITTI_PATH = None
```

### **Test 3: BDD100K Dataset (Copy After Test 2)**

```python
# === Test BDD100K Dataset ===
print("\n🚗 Testing BDD100K Dataset")

if BDD_PATH:
    try:
        from data import BDD100KSegmentationDataset
        
        # Create BDD100K dataset
        bdd_dataset = BDD100KSegmentationDataset(
            root_dir=BDD_PATH,
            split='train',
            image_size=(256, 256),
            is_training=True
        )
        
        print(f"✅ BDD100K Dataset: {len(bdd_dataset)} samples")
        print(f"   Classes: {len(bdd_dataset.BDD100K_CLASSES)}")
        print(f"   Sample classes: {list(bdd_dataset.BDD100K_CLASSES.items())[:5]}")
        
        if len(bdd_dataset) > 0:
            # Test loading a sample
            image, mask = bdd_dataset[0]
            print(f"✅ Sample loaded:")
            print(f"   Image shape: {image.shape}")
            print(f"   Mask shape: {mask.shape}")
            print(f"   Image dtype: {image.dtype}")
            print(f"   Mask dtype: {mask.dtype}")
            
            # Check value ranges
            if hasattr(image, 'min'):
                print(f"   Image range: [{image.min():.3f}, {image.max():.3f}]")
            if hasattr(mask, 'unique'):
                unique_classes = mask.unique() if hasattr(mask, 'unique') else np.unique(mask)
                print(f"   Classes in sample: {unique_classes}")
        
    except Exception as e:
        print(f"❌ BDD100K error: {e}")
else:
    print("❌ No BDD100K path available")
```

### **Test 4: KITTI Dataset (Copy After Test 3)**

```python
# === Test KITTI Dataset ===
print("\n🚗 Testing KITTI Dataset")

if KITTI_PATH:
    try:
        from data import KITTISegmentationDataset, KITTISequenceDataset
        
        # Test single frame dataset
        kitti_dataset = KITTISegmentationDataset(
            root_dir=KITTI_PATH,
            split='train',
            image_size=(256, 256)
        )
        
        print(f"✅ KITTI Dataset: {len(kitti_dataset)} samples")
        
        if len(kitti_dataset) > 0:
            image, mask = kitti_dataset[0]
            print(f"✅ KITTI sample:")
            print(f"   Image: {image.shape}, {image.dtype}")
            print(f"   Mask: {mask.shape}, {mask.dtype}")
        
        # Test sequence dataset
        seq_dataset = KITTISequenceDataset(
            root_dir=KITTI_PATH,
            sequence_length=3,
            image_size=(128, 128)
        )
        
        print(f"✅ KITTI Sequence Dataset: {len(seq_dataset)} sequences")
        
        if len(seq_dataset) > 0:
            images, masks = seq_dataset[0]
            print(f"✅ Sequence sample: {len(images)} images, {len(masks)} masks")
        
    except Exception as e:
        print(f"❌ KITTI error: {e}")
else:
    print("❌ No KITTI path available")
```

### **Test 5: Dataset Analysis (Copy After Test 4)**

```python
# === Test Dataset Analysis Tools ===
print("\n📊 Testing Dataset Analysis")

try:
    from data import DatasetPreprocessor, DatasetValidator
    
    if BDD_PATH:
        from data import BDD100KSegmentationDataset
        dataset = BDD100KSegmentationDataset(BDD_PATH, image_size=(128, 128))
        
        if len(dataset) > 0:
            print("🔍 Analyzing BDD100K dataset...")
            
            # Analyze dataset statistics
            stats = DatasetPreprocessor.analyze_dataset(dataset, num_samples=3)
            print(f"✅ Analysis complete:")
            print(f"   Total samples: {stats.get('total_samples', 0)}")
            print(f"   Classes found: {len(stats.get('class_distribution', {}))}")
            
            # Validate dataset
            print("\n🔍 Validating dataset...")
            report = DatasetValidator.validate_dataset(dataset)
            DatasetValidator.print_validation_report(report)
    
except Exception as e:
    print(f"❌ Analysis error: {e}")
```

### **Test 6: DataLoader Creation (Copy After Test 5)**

```python
# === Test DataLoader Creation ===
print("\n🔄 Testing DataLoader Creation")

try:
    import torch
    from data import create_bdd100k_dataloaders, create_kitti_dataloaders
    
    if BDD_PATH:
        print("Creating BDD100K dataloaders...")
        train_loader, val_loader = create_bdd100k_dataloaders(
            root_dir=BDD_PATH,
            batch_size=2,
            image_size=(128, 128),
            num_workers=0  # Use 0 for iPython testing
        )
        
        print(f"✅ BDD100K DataLoaders:")
        print(f"   Train: {len(train_loader)} batches")
        print(f"   Val: {len(val_loader)} batches")
        
        # Test batch loading
        if len(train_loader) > 0:
            batch_images, batch_masks = next(iter(train_loader))
            print(f"✅ Batch loaded:")
            print(f"   Images: {batch_images.shape}")
            print(f"   Masks: {batch_masks.shape}")
            print(f"   Image range: [{batch_images.min():.3f}, {batch_images.max():.3f}]")
    
    if KITTI_PATH:
        print("\nCreating KITTI dataloaders...")
        train_loader, val_loader = create_kitti_dataloaders(
            root_dir=KITTI_PATH,
            batch_size=2,
            image_size=(128, 128),
            num_workers=0
        )
        
        print(f"✅ KITTI DataLoaders created")
    
except Exception as e:
    print(f"❌ DataLoader error: {e}")
```

### **Test 7: Integration with U-Net (Copy Last)**

```python
# === Test Integration with U-Net ===
print("\n🧠 Testing U-Net Integration")

try:
    from models import UNet
    import torch
    
    # Create U-Net model
    model = UNet(n_channels=3, n_classes=19)  # 19 classes for BDD100K/KITTI
    model.eval()
    print("✅ U-Net model created for 19 classes")
    
    if BDD_PATH:
        from data import BDD100KSegmentationDataset
        dataset = BDD100KSegmentationDataset(BDD_PATH, image_size=(256, 256))
        
        if len(dataset) > 0:
            # Test model with dataset sample
            image, mask = dataset[0]
            
            if torch.is_tensor(image):
                # Add batch dimension
                input_batch = image.unsqueeze(0)  # (1, 3, 256, 256)
                
                # Run inference
                with torch.no_grad():
                    output = model(input_batch)
                    predicted_mask = torch.argmax(output, dim=1)
                
                print(f"✅ U-Net inference successful:")
                print(f"   Input: {input_batch.shape}")
                print(f"   Output: {output.shape}")
                print(f"   Predicted mask: {predicted_mask.shape}")
                print(f"   Ground truth classes: {torch.unique(mask)}")
                print(f"   Predicted classes: {torch.unique(predicted_mask)}")
    
except Exception as e:
    print(f"❌ U-Net integration error: {e}")

print("\n🎉 Dataset testing complete!")
print("\n📋 Summary:")
print("✅ BDD100K and KITTI dataset classes created")
print("✅ Dataset utilities for analysis and validation")
print("✅ DataLoader creation functions")
print("✅ Integration with U-Net model")
print("✅ Sample datasets for testing without real data")
```

## 🎯 **Complete Test Sequence (All-in-One)**

```python
# === COMPLETE DATASET TEST - COPY ALL OF THIS ===
print("🚀 Complete Dataset Module Test")

# Step 1: Test imports
from data import list_available_datasets, get_dataset_info, create_sample_dataset_for_testing
print("✅ All imports successful")

# Step 2: Show available datasets
print("\n📊 Available Datasets:")
datasets = list_available_datasets()

# Step 3: Create sample data
print("\n🏗️  Creating sample datasets...")
bdd_path, kitti_path = create_sample_dataset_for_testing("complete_test")

# Step 4: Test BDD100K
if bdd_path:
    from data import BDD100KSegmentationDataset
    bdd_dataset = BDD100KSegmentationDataset(str(bdd_path), image_size=(224, 224))
    print(f"✅ BDD100K: {len(bdd_dataset)} samples")
    
    if len(bdd_dataset) > 0:
        image, mask = bdd_dataset[0]
        print(f"   Sample: {image.shape}, {mask.shape}")

# Step 5: Test KITTI
if kitti_path:
    from data import KITTISegmentationDataset
    kitti_dataset = KITTISegmentationDataset(str(kitti_path), image_size=(224, 224))
    print(f"✅ KITTI: {len(kitti_dataset)} samples")

# Step 6: Test with U-Net
try:
    from models import UNet
    import torch
    
    model = UNet(n_channels=3, n_classes=19)
    
    if len(bdd_dataset) > 0:
        image, mask = bdd_dataset[0]
        input_batch = image.unsqueeze(0)
        
        with torch.no_grad():
            output = model(input_batch)
        
        print(f"✅ U-Net integration: {input_batch.shape} → {output.shape}")

except Exception as e:
    print(f"⚠️  U-Net test: {e}")

print("\n🎉 Complete test finished!")
print("Your data module is ready for BDD100K and KITTI datasets!")
```

## 📋 **What You Get**

### **BDD100K Dataset Features:**
- ✅ **19 semantic classes** for autonomous driving
- ✅ **Driving-specific augmentations** (fog, rain, shadow)
- ✅ **Class weights** for handling imbalance
- ✅ **Visualization tools** with proper color mapping
- ✅ **Flexible directory structure** support

### **KITTI Dataset Features:**
- ✅ **19 semantic classes** (Cityscapes format)
- ✅ **Sequence support** for video-like processing
- ✅ **Multiple directory structures** supported
- ✅ **Temporal processing** capabilities

### **Dataset Utilities:**
- ✅ **Download helpers** (when requests available)
- ✅ **Preprocessing tools** for data preparation
- ✅ **Analysis functions** for dataset statistics
- ✅ **Validation tools** for checking data integrity
- ✅ **Train/val splitting** utilities

## 🎯 **To Use with Real Data:**

### **BDD100K:**
1. Download from: https://bdd-data.berkeley.edu/
2. Run: `from data import setup_bdd100k_directory; setup_bdd100k_directory()`
3. Extract data to created structure
4. Use: `from data import create_bdd100k_dataloaders`

### **KITTI:**
1. Download from: http://www.cvlibs.net/datasets/kitti/
2. Run: `from data import setup_kitti_directory; setup_kitti_directory()`
3. Extract data to created structure  
4. Use: `from data import create_kitti_dataloaders`

**Copy the "Complete Test Sequence" above into your iPython session to test everything at once!** 🚀
