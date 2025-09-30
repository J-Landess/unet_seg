#!/usr/bin/env python3
"""
Simple BDD100K training using the working training framework
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import yaml
import time
from tqdm import tqdm
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def create_simple_unet_for_bdd100k():
    """Create simple U-Net for BDD100K (19 classes)"""
    
    class SimpleUNet(nn.Module):
        def __init__(self, in_channels=3, num_classes=19):
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
    
    return SimpleUNet()

def create_bdd100k_dummy_dataset(num_samples=20, image_size=(256, 256)):
    """Create dummy dataset with BDD100K-like classes"""
    
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
            
            # Add vegetation
            for _ in range(np.random.randint(2, 5)):
                veg_h = np.random.randint(10, 30)
                veg_w = np.random.randint(10, 30)
                veg_y = np.random.randint(h//3, 2*h//3)
                veg_x = np.random.randint(0, w - veg_w)
                mask[veg_y:veg_y+veg_h, veg_x:veg_x+veg_w] = 8  # vegetation
            
            # Convert to tensors
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()
            
            return image, mask
    
    return BDD100KDummyDataset(num_samples, image_size)

def train_bdd100k_simple(
    num_epochs: int = 5,
    batch_size: int = 2,
    learning_rate: float = 0.001
):
    """Train simple U-Net on BDD100K-like data"""
    
    print("🚗 Training Simple U-Net on BDD100K-like Data")
    print("=" * 50)
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print()
    
    # Set device
    device = torch.device('cpu')
    print(f"📱 Device: {device}")
    
    # Create model
    model = create_simple_unet_for_bdd100k().to(device)
    print(f"🧠 Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create datasets
    train_dataset = create_bdd100k_dummy_dataset(num_samples=20, image_size=(256, 256))
    val_dataset = create_bdd100k_dummy_dataset(num_samples=10, image_size=(256, 256))
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"📊 Train samples: {len(train_dataset)}")
    print(f"📊 Val samples: {len(val_dataset)}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Training loop
    print(f"\n🎯 Training for {num_epochs} epochs...")
    print("-" * 50)
    
    best_loss = float('inf')
    training_history = {
        'train_losses': [],
        'val_losses': [],
        'epochs': []
    }
    
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
        
        training_history['train_losses'].append(avg_train_loss)
        training_history['val_losses'].append(avg_val_loss)
        training_history['epochs'].append(epoch + 1)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            model_path = "checkpoints/bdd100k_simple_best.pth"
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print(f"  💾 New best model saved! (Val Loss: {avg_val_loss:.4f})")
        
        print("-" * 50)
    
    print(f"\n✅ BDD100K training completed!")
    print(f"🏆 Best validation loss: {best_loss:.4f}")
    print(f"💾 Best model saved to: checkpoints/bdd100k_simple_best.pth")
    
    return {
        'model': model,
        'training_history': training_history,
        'best_loss': best_loss,
        'model_path': "checkpoints/bdd100k_simple_best.pth"
    }

def test_bdd100k_model_on_video(
    model_path: str = "checkpoints/bdd100k_simple_best.pth",
    video_path: str = "video_processing/data3_users_yoavnavon_clips_dqa_glued_sv_vehicle_day_FRONT_RIGHT_GENERIC_1x.mp4",
    output_dir: str = "bdd100k_video_inference_output"
):
    """Test BDD100K trained model on video"""
    
    print("🎬 Testing BDD100K Model on Video")
    print("=" * 40)
    
    from tests.test_video_inference import VideoInferenceTester
    
    # Create custom model loader for BDD100K
    def load_bdd100k_model():
        model = create_simple_unet_for_bdd100k()
        if model and os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
            return model
        return None
    
    # Test with custom model
    tester = VideoInferenceTester()
    
    # Override model loading
    original_load_model = tester._load_model
    def custom_load_model(model_type, model_path, encoder_name):
        if model_type == "trained" and "bdd100k" in model_path:
            return load_bdd100k_model()
        return original_load_model(model_type, model_path, encoder_name)
    
    tester._load_model = custom_load_model
    
    # Run inference
    results = tester.test_trained_model_inference(
        video_path=video_path,
        model_path=model_path,
        output_dir=output_dir,
        frame_skip=30,
        max_frames=5
    )
    
    return results

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Simple U-Net on BDD100K-like Data")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--test-video", action="store_true", help="Test on video after training")
    
    args = parser.parse_args()
    
    # Train model
    results = train_bdd100k_simple(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
    
    if results and args.test_video:
        # Test on video
        test_bdd100k_model_on_video(
            model_path=results['model_path']
        )

if __name__ == "__main__":
    main()
