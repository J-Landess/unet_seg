#!/usr/bin/env python3
"""
Training testing module for U-Net project

This module consolidates tests for:
- Training pipeline validation
- Model architecture testing
- BDD100K training
- Simple training tests
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import cv2
import yaml
from pathlib import Path
import time
from typing import Dict, Any, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from data.bdd100k_dataset import BDD100KSegmentationDataset
from data.dataset_utils import create_sample_dataset_for_testing


class TrainingTester:
    """Test training functionality and model validation"""
    
    def __init__(self, output_dir: str = "test_outputs/training_tests"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
    
    def test_simple_training(
        self,
        num_epochs: int = 3,
        batch_size: int = 2,
        learning_rate: float = 0.001,
        image_size: Tuple[int, int] = (256, 256)
    ) -> Dict[str, Any]:
        """Test simple training pipeline"""
        
        print("🚀 Testing Simple Training Pipeline")
        print("=" * 40)
        
        results = {
            'model_created': False,
            'dataset_created': False,
            'training_completed': False,
            'final_loss': float('inf'),
            'training_time': 0,
            'errors': []
        }
        
        try:
            # Create simple U-Net model
            print("Creating U-Net model...")
            model = self._create_simple_unet()
            print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
            results['model_created'] = True
            
            # Create simple dataset
            print("Creating training dataset...")
            train_dataset, val_dataset = self._create_simple_dataset(image_size)
            print(f"✅ Dataset created: {len(train_dataset)} train, {len(val_dataset)} val samples")
            results['dataset_created'] = True
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # Setup training
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            # Training loop
            print(f"Starting training for {num_epochs} epochs...")
            start_time = time.time()
            
            for epoch in range(num_epochs):
                # Training
                model.train()
                train_loss = 0.0
                for batch_idx, (data, target) in enumerate(train_loader):
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                # Validation
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for data, target in val_loader:
                        output = model(data)
                        loss = criterion(output, target)
                        val_loss += loss.item()
                
                train_loss /= len(train_loader)
                val_loss /= len(val_loader)
                
                print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            training_time = time.time() - start_time
            results['training_completed'] = True
            results['final_loss'] = val_loss
            results['training_time'] = training_time
            
            print(f"✅ Training completed in {training_time:.2f} seconds")
            print(f"   Final validation loss: {val_loss:.4f}")
            
        except Exception as e:
            print(f"❌ Simple training failed: {e}")
            results['errors'].append(str(e))
        
        return results
    
    def test_bdd100k_training(
        self,
        num_epochs: int = 5,
        batch_size: int = 2,
        learning_rate: float = 0.001,
        image_size: Tuple[int, int] = (256, 256)
    ) -> Dict[str, Any]:
        """Test BDD100K training pipeline"""
        
        print("🏗️ Testing BDD100K Training Pipeline")
        print("=" * 40)
        
        results = {
            'dataset_created': False,
            'model_created': False,
            'training_completed': False,
            'final_loss': float('inf'),
            'training_time': 0,
            'errors': []
        }
        
        try:
            # Create BDD100K dataset
            print("Creating BDD100K dataset...")
            dataset = BDD100KSegmentationDataset(
                data_dir="data/sample_bdd100k",
                split="train",
                image_size=image_size,
                augmentation=True
            )
            
            if len(dataset) == 0:
                print("⚠️ No BDD100K data found, creating dummy dataset...")
                dataset = self._create_bdd100k_dummy_dataset(num_samples=20, image_size=image_size)
            
            print(f"✅ BDD100K dataset created with {len(dataset)} samples")
            results['dataset_created'] = True
            
            # Split dataset
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            
            # Create model
            print("Creating U-Net model...")
            model = self._create_simple_unet(num_classes=19)  # BDD100K has 19 classes
            print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
            results['model_created'] = True
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # Setup training
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            # Training loop
            print(f"Starting BDD100K training for {num_epochs} epochs...")
            start_time = time.time()
            best_val_loss = float('inf')
            
            for epoch in range(num_epochs):
                # Training
                model.train()
                train_loss = 0.0
                for batch_idx, (data, target) in enumerate(train_loader):
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                # Validation
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for data, target in val_loader:
                        output = model(data)
                        loss = criterion(output, target)
                        val_loss += loss.item()
                
                train_loss /= len(train_loader)
                val_loss /= len(val_loader)
                
                print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    # Save best model
                    model_path = self.output_dir / "best_bdd100k_model.pth"
                    torch.save(model.state_dict(), model_path)
            
            training_time = time.time() - start_time
            results['training_completed'] = True
            results['final_loss'] = best_val_loss
            results['training_time'] = training_time
            
            print(f"✅ BDD100K training completed in {training_time:.2f} seconds")
            print(f"   Best validation loss: {best_val_loss:.4f}")
            print(f"   Model saved to: {self.output_dir / 'best_bdd100k_model.pth'}")
            
        except Exception as e:
            print(f"❌ BDD100K training failed: {e}")
            results['errors'].append(str(e))
        
        return results
    
    def test_model_architecture(self) -> Dict[str, Any]:
        """Test model architecture and forward pass"""
        
        print("🏗️ Testing Model Architecture")
        print("=" * 40)
        
        results = {
            'model_created': False,
            'forward_pass': False,
            'parameter_count': 0,
            'model_size_mb': 0,
            'errors': []
        }
        
        try:
            # Create model
            model = self._create_simple_unet()
            results['model_created'] = True
            
            # Count parameters
            param_count = sum(p.numel() for p in model.parameters())
            results['parameter_count'] = param_count
            print(f"✅ Model created with {param_count:,} parameters")
            
            # Calculate model size
            model_size = sum(p.numel() * p.element_size() for p in model.parameters())
            model_size_mb = model_size / (1024 * 1024)
            results['model_size_mb'] = model_size_mb
            print(f"   Model size: {model_size_mb:.2f} MB")
            
            # Test forward pass
            print("Testing forward pass...")
            model.eval()
            with torch.no_grad():
                # Create dummy input
                dummy_input = torch.randn(1, 3, 256, 256)
                output = model(dummy_input)
                
                print(f"   Input shape: {dummy_input.shape}")
                print(f"   Output shape: {output.shape}")
                print(f"   Output classes: {output.shape[1]}")
                
                results['forward_pass'] = True
                print("✅ Forward pass successful")
            
        except Exception as e:
            print(f"❌ Model architecture test failed: {e}")
            results['errors'].append(str(e))
        
        return results
    
    def test_training_config(self, config_path: str = "config/config_cpu.yaml") -> Dict[str, Any]:
        """Test training configuration loading"""
        
        print("⚙️ Testing Training Configuration")
        print("=" * 40)
        
        results = {
            'config_loaded': False,
            'config_valid': False,
            'required_keys': [],
            'errors': []
        }
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                print(f"✅ Config loaded from {config_path}")
                results['config_loaded'] = True
                
                # Check required keys
                required_keys = ['model', 'training', 'data']
                missing_keys = [key for key in required_keys if key not in config]
                
                if not missing_keys:
                    results['config_valid'] = True
                    print("✅ Configuration is valid")
                    
                    # Print key settings
                    if 'training' in config:
                        print(f"   Learning rate: {config['training'].get('learning_rate', 'N/A')}")
                        print(f"   Batch size: {config['training'].get('batch_size', 'N/A')}")
                        print(f"   Epochs: {config['training'].get('num_epochs', 'N/A')}")
                    
                    if 'model' in config:
                        print(f"   Model classes: {config['model'].get('num_classes', 'N/A')}")
                        print(f"   Input size: {config['model'].get('input_size', 'N/A')}")
                
                else:
                    print(f"❌ Missing required keys: {missing_keys}")
                    results['errors'].append(f"Missing keys: {missing_keys}")
                
                results['required_keys'] = required_keys
                
            else:
                print(f"❌ Config file not found: {config_path}")
                results['errors'].append(f"Config file not found: {config_path}")
        
        except Exception as e:
            print(f"❌ Config loading failed: {e}")
            results['errors'].append(str(e))
        
        return results
    
    def _create_simple_unet(self, in_channels: int = 3, num_classes: int = 21) -> nn.Module:
        """Create a simple U-Net model for testing"""
        
        class SimpleUNet(nn.Module):
            def __init__(self, in_channels=3, num_classes=21):
                super().__init__()
                self.enc1 = self.conv_block(in_channels, 64)
                self.enc2 = self.conv_block(64, 128)
                self.enc3 = self.conv_block(128, 256)
                self.enc4 = self.conv_block(256, 512)
                
                self.bottleneck = self.conv_block(512, 1024)
                
                self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
                self.dec4 = self.conv_block(1024, 512)
                
                self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
                self.dec3 = self.conv_block(512, 256)
                
                self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
                self.dec2 = self.conv_block(256, 128)
                
                self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
                self.dec1 = self.conv_block(128, 64)
                
                self.final = nn.Conv2d(64, num_classes, kernel_size=1)
            
            def conv_block(self, in_channels, out_channels):
                return nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            
            def forward(self, x):
                # Encoder
                enc1 = self.enc1(x)
                enc2 = self.enc2(torch.max_pool2d(enc1, 2))
                enc3 = self.enc3(torch.max_pool2d(enc2, 2))
                enc4 = self.enc4(torch.max_pool2d(enc3, 2))
                
                # Bottleneck
                bottleneck = self.bottleneck(torch.max_pool2d(enc4, 2))
                
                # Decoder
                dec4 = self.upconv4(bottleneck)
                dec4 = torch.cat((dec4, enc4), dim=1)
                dec4 = self.dec4(dec4)
                
                dec3 = self.upconv3(dec4)
                dec3 = torch.cat((dec3, enc3), dim=1)
                dec3 = self.dec3(dec3)
                
                dec2 = self.upconv2(dec3)
                dec2 = torch.cat((dec2, enc2), dim=1)
                dec2 = self.dec2(dec2)
                
                dec1 = self.upconv1(dec2)
                dec1 = torch.cat((dec1, enc1), dim=1)
                dec1 = self.dec1(dec1)
                
                return self.final(dec1)
        
        return SimpleUNet(in_channels, num_classes)
    
    def _create_simple_dataset(self, image_size: Tuple[int, int]) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
        """Create a simple dataset for testing"""
        
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, num_samples: int, image_size: Tuple[int, int]):
                self.num_samples = num_samples
                self.image_size = image_size
            
            def __len__(self):
                return self.num_samples
            
            def __getitem__(self, idx):
                h, w = self.image_size
                
                # Create random image
                image = torch.randn(3, h, w)
                
                # Create random mask
                mask = torch.randint(0, 21, (h, w))
                
                return image, mask
        
        # Create train and validation datasets
        train_dataset = SimpleDataset(20, image_size)
        val_dataset = SimpleDataset(5, image_size)
        
        return train_dataset, val_dataset
    
    def _create_bdd100k_dummy_dataset(self, num_samples: int = 20, image_size: Tuple[int, int] = (256, 256)):
        """Create dummy BDD100K dataset for testing"""
        
        class BDD100KDummyDataset:
            def __init__(self, num_samples, image_size):
                self.num_samples = num_samples
                self.image_size = image_size
                
                # BDD100K classes (simplified)
                self.classes = {
                    0: 'road', 1: 'sidewalk', 2: 'building', 3: 'wall', 4: 'fence',
                    5: 'pole', 6: 'traffic_light', 7: 'traffic_sign', 8: 'vegetation',
                    9: 'terrain', 10: 'sky', 11: 'person', 12: 'rider', 13: 'car',
                    14: 'truck', 15: 'bus', 16: 'train', 17: 'motorcycle', 18: 'bicycle'
                }
            
            def __len__(self):
                return self.num_samples
            
            def __getitem__(self, idx):
                h, w = self.image_size
                
                # Create driving scene-like image
                image = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
                
                # Create more realistic driving scene
                # Sky (top portion)
                image[:h//3, :] = [135, 206, 235]  # Sky blue
                
                # Road (bottom portion)
                image[2*h//3:, :] = [105, 105, 105]  # Dim gray
                
                # Add some variation
                variation = np.random.randint(-20, 20, (h, w, 3), dtype=np.int16)
                image = image.astype(np.int16) + variation
                image = np.clip(image, 0, 255).astype(np.uint8)
                
                # Create segmentation mask
                mask = np.zeros((h, w), dtype=np.uint8)
                
                # Sky
                mask[:h//3, :] = 10  # sky
                
                # Road
                mask[2*h//3:, :] = 0  # road
                
                # Add some cars
                for _ in range(np.random.randint(1, 4)):
                    car_h = np.random.randint(20, 40)
                    car_w = np.random.randint(30, 60)
                    car_y = np.random.randint(h//2, h - car_h)
                    car_x = np.random.randint(0, w - car_w)
                    mask[car_y:car_y+car_h, car_x:car_x+car_w] = 13  # car
                
                # Add buildings
                for _ in range(np.random.randint(1, 3)):
                    building_h = np.random.randint(40, h//2)
                    building_w = np.random.randint(50, 100)
                    building_y = np.random.randint(h//3, h//2)
                    building_x = np.random.randint(0, w - building_w)
                    mask[building_y:building_y+building_h, building_x:building_x+building_w] = 2  # building
                
                return torch.from_numpy(image).permute(2, 0, 1).float() / 255.0, torch.from_numpy(mask).long()
        
        return BDD100KDummyDataset(num_samples, image_size)
    
    def run_all_training_tests(self) -> Dict[str, Any]:
        """Run all training tests"""
        
        print("🧪 Running All Training Tests")
        print("=" * 50)
        
        results = {}
        
        # Test model architecture
        results['architecture'] = self.test_model_architecture()
        
        # Test training configuration
        results['config'] = self.test_training_config()
        
        # Test simple training
        results['simple_training'] = self.test_simple_training()
        
        # Test BDD100K training
        results['bdd100k_training'] = self.test_bdd100k_training()
        
        # Print summary
        print(f"\n📋 Training Test Summary")
        print("=" * 50)
        for test_name, test_results in results.items():
            if test_name in ['architecture', 'config']:
                status = "✅" if test_results.get('config_valid' if test_name == 'config' else 'forward_pass', False) else "❌"
            else:
                status = "✅" if test_results.get('training_completed', False) else "❌"
            print(f"{test_name}: {status}")
        
        return results


def main():
    """Run training tests"""
    tester = TrainingTester()
    results = tester.run_all_training_tests()
    
    # Save results
    results_file = tester.output_dir / "training_test_results.txt"
    with open(results_file, 'w') as f:
        f.write("Training Test Results\n")
        f.write("=" * 30 + "\n\n")
        
        for test_name, test_results in results.items():
            f.write(f"{test_name.upper()}:\n")
            for key, value in test_results.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
    
    print(f"\n📄 Results saved to: {results_file}")


if __name__ == "__main__":
    main()
