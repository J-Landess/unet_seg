#!/usr/bin/env python3
"""
Simple training test script that works with the current codebase structure
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

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from video_processing.frame_iterator import VideoFrameIterator

class SimpleUNet(nn.Module):
    """Simple U-Net implementation for testing"""
    
    def __init__(self, in_channels=3, num_classes=21):
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

class SimpleDataset:
    """Simple dataset for testing"""
    
    def __init__(self, data_dir, image_size=(256, 256), num_samples=50):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.num_samples = num_samples
        
        # Create dummy data
        self.samples = []
        for i in range(num_samples):
            self.samples.append({
                'image': f'dummy_image_{i}.jpg',
                'mask': f'dummy_mask_{i}.png'
            })
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Create dummy image and mask
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

def train_model(config_path="config/config_cpu.yaml"):
    """Train the model with the given configuration"""
    
    print("🚀 Starting U-Net Training (CPU)")
    print("=" * 50)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device('cpu')
    print(f"📱 Device: {device}")
    
    # Create model
    model = SimpleUNet(
        in_channels=config['model']['input_channels'],
        num_classes=config['model']['num_classes']
    ).to(device)
    
    print(f"🧠 Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create datasets
    train_dataset = SimpleDataset(
        data_dir="data",
        image_size=tuple(config['data']['image_size']),
        num_samples=50
    )
    
    val_dataset = SimpleDataset(
        data_dir="data",
        image_size=tuple(config['data']['image_size']),
        num_samples=20
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )
    
    print(f"📊 Train samples: {len(train_dataset)}")
    print(f"📊 Val samples: {len(val_dataset)}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(config['training']['learning_rate']),
        weight_decay=float(config['training']['weight_decay'])
    )
    
    # Training loop
    num_epochs = config['training']['num_epochs']
    best_loss = float('inf')
    
    print(f"\n🎯 Training for {num_epochs} epochs...")
    print("-" * 50)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch_idx, (images, masks) in enumerate(train_pbar):
            images, masks = images.to(device), masks.to(device)
            
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
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                val_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), 'checkpoints/best_model.pth')
            print(f"  💾 New best model saved! (Val Loss: {avg_val_loss:.4f})")
        
        print("-" * 50)
    
    print(f"\n✅ Training completed!")
    print(f"🏆 Best validation loss: {best_loss:.4f}")
    print(f"💾 Best model saved to: checkpoints/best_model.pth")

def main():
    """Main function"""
    train_model()

if __name__ == "__main__":
    main()
