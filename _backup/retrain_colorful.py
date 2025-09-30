#!/usr/bin/env python3
"""
Retrain BDD100K model with more diverse classes for colorful output
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import cv2
import os
from pathlib import Path

def create_diverse_dummy_dataset(num_samples=50, image_size=(256, 256)):
    """Create dummy dataset with more diverse BDD100K classes"""
    
    class DiverseBDD100KDataset:
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
            
            # Create segmentation mask with more classes
            mask = np.zeros((h, w), dtype=np.uint8)
            
            # Sky (always present)
            mask[:h//3, :] = 10  # sky
            
            # Road (always present)
            mask[2*h//3:, :] = 0  # road
            
            # Sidewalk (between sky and road)
            sidewalk_y = h//3 + np.random.randint(0, h//6)
            mask[sidewalk_y:sidewalk_y+h//12, :] = 1  # sidewalk
            
            # Add buildings (always present)
            for _ in range(np.random.randint(2, 4)):
                building_h = np.random.randint(40, h//2)
                building_w = np.random.randint(50, 120)
                building_y = np.random.randint(h//3, h//2)
                building_x = np.random.randint(0, w - building_w)
                mask[building_y:building_y+building_h, building_x:building_x+building_w] = 2  # building
            
            # Add vegetation (trees) - always present
            for _ in range(np.random.randint(3, 6)):
                tree_h = np.random.randint(30, 60)
                tree_w = np.random.randint(20, 40)
                tree_y = np.random.randint(h//3, 2*h//3)
                tree_x = np.random.randint(0, w - tree_w)
                # Create tree shape
                for i in range(tree_h):
                    width = max(5, tree_w - (i * tree_w // tree_h))
                    start_x = tree_x + (tree_w - width) // 2
                    mask[tree_y+i:tree_y+i+1, start_x:start_x+width] = 8  # vegetation
            
            # Add cars - always present
            for _ in range(np.random.randint(2, 5)):
                car_h = np.random.randint(20, 40)
                car_w = np.random.randint(30, 60)
                car_y = np.random.randint(h//2, h - car_h)
                car_x = np.random.randint(0, w - car_w)
                mask[car_y:car_y+car_h, car_x:car_x+car_w] = 13  # car
            
            # Add traffic signs - always present
            for _ in range(np.random.randint(2, 4)):
                sign_h = np.random.randint(15, 25)
                sign_w = np.random.randint(10, 20)
                sign_y = np.random.randint(h//3, 2*h//3)
                sign_x = np.random.randint(0, w - sign_w)
                mask[sign_y:sign_y+sign_h, sign_x:sign_x+sign_w] = 7  # traffic_sign
            
            # Add poles - always present
            for _ in range(np.random.randint(2, 5)):
                pole_h = np.random.randint(40, 80)
                pole_w = 3
                pole_y = np.random.randint(h//3, 2*h//3)
                pole_x = np.random.randint(0, w - pole_w)
                mask[pole_y:pole_y+pole_h, pole_x:pole_x+pole_w] = 5  # pole
            
            # Add people - sometimes present
            if np.random.random() > 0.3:
                for _ in range(np.random.randint(1, 3)):
                    person_h = np.random.randint(15, 25)
                    person_w = np.random.randint(8, 15)
                    person_y = np.random.randint(h//2, h - person_h)
                    person_x = np.random.randint(0, w - person_w)
                    mask[person_y:person_y+person_h, person_x:person_x+person_w] = 11  # person
            
            # Add traffic lights - sometimes present
            if np.random.random() > 0.5:
                for _ in range(np.random.randint(1, 3)):
                    light_h = np.random.randint(20, 30)
                    light_w = np.random.randint(8, 12)
                    light_y = np.random.randint(h//3, 2*h//3)
                    light_x = np.random.randint(0, w - light_w)
                    mask[light_y:light_y+light_h, light_x:light_x+light_w] = 6  # traffic_light
            
            # Add terrain patches
            for _ in range(np.random.randint(1, 3)):
                terrain_h = np.random.randint(20, 40)
                terrain_w = np.random.randint(30, 60)
                terrain_y = np.random.randint(h//3, 2*h//3)
                terrain_x = np.random.randint(0, w - terrain_w)
                mask[terrain_y:terrain_y+terrain_h, terrain_x:terrain_x+terrain_w] = 9  # terrain
            
            return torch.from_numpy(image).permute(2, 0, 1).float() / 255.0, torch.from_numpy(mask).long()
    
    return DiverseBDD100KDataset(num_samples, image_size)

def train_diverse_model(num_epochs=10, batch_size=4, learning_rate=0.001):
    """Train model with diverse dataset"""
    
    print("🎨 Training Diverse BDD100K Model")
    print("=" * 50)
    
    # Create dataset
    dataset = create_diverse_dummy_dataset(num_samples=100, image_size=(256, 256))
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Model (same as before)
    class SimpleUNet(nn.Module):
        def __init__(self, in_channels=3, num_classes=19):
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
    
    # Initialize model
    model = SimpleUNet(in_channels=3, num_classes=19)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"📱 Device: {device}")
    print(f"🧠 Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"📊 Train samples: {len(train_dataset)}")
    print(f"📊 Val samples: {len(val_dataset)}")
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
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
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), 'checkpoints/diverse_bdd100k_best.pth')
            print(f"  💾 New best model saved! (Val Loss: {val_loss:.4f})")
        print("-" * 50)
    
    print(f"\n✅ Diverse training completed!")
    print(f"🏆 Best validation loss: {best_val_loss:.4f}")
    print(f"💾 Best model saved to: checkpoints/diverse_bdd100k_best.pth")
    
    return {
        'model_path': 'checkpoints/diverse_bdd100k_best.pth',
        'best_val_loss': best_val_loss
    }

if __name__ == "__main__":
    results = train_diverse_model(num_epochs=15, batch_size=4)
    print(f"\n🎯 Training complete! Model saved to: {results['model_path']}")
