#!/usr/bin/env python3
"""
Training testing module
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import cv2
from pathlib import Path
import yaml
import time
from tqdm import tqdm
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

class TrainingTester:
    """Test training functionality"""
    
    def __init__(self, output_dir: str = "test_outputs/training"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def test_simple_training(
        self,
        config_path: str = "config/config_cpu.yaml",
        num_epochs: int = 5,
        batch_size: int = 2
    ) -> Dict[str, Any]:
        """Test simple training with dummy data"""
        
        print("🚀 Testing Simple Training")
        print("=" * 40)
        
        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Override config for testing
        config['training']['num_epochs'] = num_epochs
        config['training']['batch_size'] = batch_size
        
        # Set device
        device = torch.device('cpu')
        print(f"📱 Device: {device}")
        
        # Create model
        model = self._create_simple_unet(
            in_channels=config['model']['input_channels'],
            num_classes=config['model']['num_classes']
        ).to(device)
        
        print(f"🧠 Model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Create datasets
        train_dataset = self._create_dummy_dataset(
            num_samples=20,  # Smaller for testing
            image_size=tuple(config['data']['image_size'])
        )
        
        val_dataset = self._create_dummy_dataset(
            num_samples=10,  # Smaller for testing
            image_size=tuple(config['data']['image_size'])
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=0  # Use 0 for testing
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=0
        )
        
        print(f"📊 Train samples: {len(train_dataset)}")
        print(f"📊 Val samples: {len(val_dataset)}")
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=float(config['training']['learning_rate']),
            weight_decay=float(config['training']['weight_decay'])
        )
        
        # Training loop
        training_results = self._run_training_loop(
            model, train_loader, val_loader, criterion, optimizer, num_epochs
        )
        
        # Save model
        model_path = self.output_dir / "test_model.pth"
        torch.save(model.state_dict(), model_path)
        print(f"💾 Model saved to: {model_path}")
        
        return {
            "model_path": str(model_path),
            "training_results": training_results,
            "config": config
        }
    
    def test_pretrained_training(
        self,
        encoder_name: str = "vgg11",
        num_epochs: int = 3,
        batch_size: int = 2
    ) -> Dict[str, Any]:
        """Test training with pre-trained encoder"""
        
        print(f"🏗️ Testing Pre-trained Training ({encoder_name})")
        print("=" * 50)
        
        try:
            from segmentation_models_pytorch import Unet
            
            # Create pre-trained model
            model = Unet(
                encoder_name=encoder_name,
                encoder_weights="imagenet",
                classes=21,
                activation=None,
                in_channels=3
            )
            
            print(f"🧠 Pre-trained model created with {sum(p.numel() for p in model.parameters())} parameters")
            
            # Create datasets
            train_dataset = self._create_dummy_dataset(num_samples=20, image_size=(256, 256))
            val_dataset = self._create_dummy_dataset(num_samples=10, image_size=(256, 256))
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            
            # Training setup
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Training loop
            training_results = self._run_training_loop(
                model, train_loader, val_loader, criterion, optimizer, num_epochs
            )
            
            # Save model
            model_path = self.output_dir / f"pretrained_{encoder_name}_model.pth"
            torch.save(model.state_dict(), model_path)
            print(f"💾 Pre-trained model saved to: {model_path}")
            
            return {
                "model_path": str(model_path),
                "training_results": training_results,
                "encoder_name": encoder_name
            }
            
        except ImportError:
            print("❌ segmentation-models-pytorch not installed")
            return {"error": "Pre-trained models not available"}
    
    def _create_simple_unet(self, in_channels=3, num_classes=21):
        """Create simple U-Net model"""
        
        class SimpleUNet(nn.Module):
            def __init__(self, in_channels, num_classes):
                super(SimpleUNet, self).__init__()
                
                # Encoder
                self.enc1 = self.conv_block(in_channels, 64)
                self.enc2 = self.conv_block(64, 128)
                self.enc3 = self.conv_block(128, 256)
                self.enc4 = self.conv_block(256, 512)
                
                # Bottleneck
                self.bottleneck = self.conv_block(512, 1024)
                
                # Decoder
                self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
                self.dec4 = self.conv_block(1024, 512)
                
                self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
                self.dec3 = self.conv_block(512, 256)
                
                self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
                self.dec2 = self.conv_block(256, 128)
                
                self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
                self.dec1 = self.conv_block(128, 64)
                
                # Final classification
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
    
    def _create_dummy_dataset(self, num_samples=50, image_size=(256, 256)):
        """Create dummy dataset for testing"""
        
        class DummyDataset:
            def __init__(self, num_samples, image_size):
                self.num_samples = num_samples
                self.image_size = image_size
                
            def __len__(self):
                return self.num_samples
            
            def __getitem__(self, idx):
                h, w = self.image_size
                
                # Random image
                image = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
                
                # Random mask with some structure
                mask = np.zeros((h, w), dtype=np.uint8)
                
                # Add some geometric shapes
                center_x, center_y = w // 2, h // 2
                
                # Circle
                y, x = np.ogrid[:h, :w]
                circle_mask = (x - center_x)**2 + (y - center_y)**2 < (min(w, h) // 4)**2
                mask[circle_mask] = 1
                
                # Rectangle
                mask[h//4:h*3//4, w//4:w*3//4] = 2
                
                # Random noise
                noise_mask = np.random.random((h, w)) > 0.7
                mask[noise_mask] = np.random.randint(0, 21, noise_mask.sum())
                
                # Convert to tensors
                image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
                mask = torch.from_numpy(mask).long()
                
                return image, mask
        
        return DummyDataset(num_samples, image_size)
    
    def _run_training_loop(
        self, model, train_loader, val_loader, criterion, optimizer, num_epochs
    ):
        """Run training loop"""
        
        results = {
            "train_losses": [],
            "val_losses": [],
            "epochs": []
        }
        
        best_loss = float('inf')
        
        print(f"\n🎯 Training for {num_epochs} epochs...")
        print("-" * 50)
        
        for epoch in range(num_epochs):
            # Training
            model.train()
            train_loss = 0.0
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
            for batch_idx, (images, masks) in enumerate(train_pbar):
                images, masks = images.to(model.device if hasattr(model, 'device') else 'cpu'), masks.to(model.device if hasattr(model, 'device') else 'cpu')
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
            # Validation
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
                for images, masks in val_pbar:
                    images, masks = images.to(model.device if hasattr(model, 'device') else 'cpu'), masks.to(model.device if hasattr(model, 'device') else 'cpu')
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    val_loss += loss.item()
                    val_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
            # Calculate average losses
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            results["train_losses"].append(avg_train_loss)
            results["val_losses"].append(avg_val_loss)
            results["epochs"].append(epoch + 1)
            
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss: {avg_val_loss:.4f}")
            
            # Save best model
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                print(f"  💾 New best model! (Val Loss: {avg_val_loss:.4f})")
            
            print("-" * 50)
        
        results["best_val_loss"] = best_loss
        return results

def main():
    """Example usage"""
    tester = TrainingTester()
    
    # Test simple training
    simple_results = tester.test_simple_training(num_epochs=3)
    
    # Test pre-trained training
    pretrained_results = tester.test_pretrained_training(encoder_name="vgg11", num_epochs=2)
    
    print(f"\n✅ Training tests completed!")
    print(f"Results saved to: {tester.output_dir}/")

if __name__ == "__main__":
    main()
