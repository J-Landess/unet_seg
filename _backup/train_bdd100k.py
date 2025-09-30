#!/usr/bin/env python3
"""
Train U-Net on BDD100K dataset for semantic segmentation
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

from data.bdd100k_dataset import BDD100KSegmentationDataset
from tests.test_training import TrainingTester

def create_bdd100k_model(pretrained=True, encoder_name="vgg11"):
    """Create model for BDD100K (19 classes)"""
    
    if pretrained:
        try:
            from segmentation_models_pytorch import Unet
            model = Unet(
                encoder_name=encoder_name,
                encoder_weights="imagenet",
                classes=19,  # BDD100K has 19 classes
                activation=None,
                in_channels=3
            )
            print(f"✅ Created pre-trained U-Net with {encoder_name} encoder")
            return model
        except ImportError:
            print("❌ segmentation-models-pytorch not installed")
            print("Run: pip install segmentation-models-pytorch")
            return None
    else:
        # Create simple U-Net from scratch
        from tests.test_training import TrainingTester
        tester = TrainingTester()
        model = tester._create_simple_unet(in_channels=3, num_classes=19)
        print("✅ Created simple U-Net from scratch")
        return model

def train_bdd100k_model(
    bdd100k_root: str,
    config_path: str = "config/config_cpu.yaml",
    pretrained: bool = True,
    encoder_name: str = "vgg11",
    num_epochs: int = 10,
    batch_size: int = 4,
    image_size: tuple = (512, 512)
):
    """Train model on BDD100K dataset"""
    
    print("🚗 Training U-Net on BDD100K Dataset")
    print("=" * 50)
    print(f"Dataset: {bdd100k_root}")
    print(f"Pre-trained: {pretrained}")
    print(f"Encoder: {encoder_name}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print()
    
    # Check if BDD100K dataset exists
    if not os.path.exists(bdd100k_root):
        print(f"❌ BDD100K dataset not found at: {bdd100k_root}")
        print("\n📥 To download BDD100K:")
        print("1. Visit: https://bdd-data.berkeley.edu/")
        print("2. Download: bdd100k_sem_seg_labels_trainval.zip")
        print("3. Extract to the specified directory")
        print("\n🔧 Or create sample data:")
        print("python3 -c \"from data.dataset_utils import create_sample_dataset_for_testing; create_sample_dataset_for_testing('bdd100k_sample')\"")
        return None
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Device: {device}")
    
    # Create model
    model = create_bdd100k_model(pretrained=pretrained, encoder_name=encoder_name)
    if model is None:
        return None
    
    model = model.to(device)
    print(f"🧠 Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create datasets
    print("\n📊 Creating datasets...")
    
    try:
        train_dataset = BDD100KSegmentationDataset(
            root_dir=bdd100k_root,
            split="train",
            image_size=image_size,
            augmentation=True
        )
        
        val_dataset = BDD100KSegmentationDataset(
            root_dir=bdd100k_root,
            split="val", 
            image_size=image_size,
            augmentation=False
        )
        
        print(f"✅ Train samples: {len(train_dataset)}")
        print(f"✅ Val samples: {len(val_dataset)}")
        
    except Exception as e:
        print(f"❌ Error creating datasets: {e}")
        print("\n💡 Try using sample data first:")
        print("python3 -c \"from data.dataset_utils import create_sample_dataset_for_testing; create_sample_dataset_for_testing('bdd100k_sample')\"")
        return None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Training setup
    criterion = nn.CrossEntropyLoss(ignore_index=255)  # Ignore background class
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
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
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            model_path = f"checkpoints/bdd100k_{encoder_name}_best.pth"
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print(f"  💾 New best model saved! (Val Loss: {avg_val_loss:.4f})")
        
        # Update learning rate
        scheduler.step()
        print("-" * 50)
    
    print(f"\n✅ BDD100K training completed!")
    print(f"🏆 Best validation loss: {best_loss:.4f}")
    print(f"💾 Best model saved to: checkpoints/bdd100k_{encoder_name}_best.pth")
    
    return {
        'model': model,
        'training_history': training_history,
        'best_loss': best_loss,
        'model_path': f"checkpoints/bdd100k_{encoder_name}_best.pth"
    }

def test_bdd100k_model_on_video(
    model_path: str,
    video_path: str = "video_processing/data3_users_yoavnavon_clips_dqa_glued_sv_vehicle_day_FRONT_RIGHT_GENERIC_1x.mp4",
    output_dir: str = "bdd100k_video_inference_output"
):
    """Test BDD100K trained model on video"""
    
    print("🎬 Testing BDD100K Model on Video")
    print("=" * 40)
    
    from tests.test_video_inference import VideoInferenceTester
    
    # Create custom model loader for BDD100K
    def load_bdd100k_model():
        model = create_bdd100k_model(pretrained=False, encoder_name="vgg11")
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
    
    parser = argparse.ArgumentParser(description="Train U-Net on BDD100K")
    parser.add_argument("--bdd100k-root", default="bdd100k", help="BDD100K dataset root directory")
    parser.add_argument("--pretrained", action="store_true", help="Use pre-trained encoder")
    parser.add_argument("--encoder", default="vgg11", help="Encoder name (vgg11, efficientnet-b0, etc.)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--test-video", action="store_true", help="Test on video after training")
    
    args = parser.parse_args()
    
    # Train model
    results = train_bdd100k_model(
        bdd100k_root=args.bdd100k_root,
        pretrained=args.pretrained,
        encoder_name=args.encoder,
        num_epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    if results and args.test_video:
        # Test on video
        test_bdd100k_model_on_video(
            model_path=results['model_path'],
            video_path="video_processing/data3_users_yoavnavon_clips_dqa_glued_sv_vehicle_day_FRONT_RIGHT_GENERIC_1x.mp4"
        )

if __name__ == "__main__":
    main()
