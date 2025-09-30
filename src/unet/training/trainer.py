"""
Training pipeline for semantic segmentation
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import wandb
from tqdm import tqdm
from typing import Dict, Any, Optional, Tuple
import numpy as np

from ..models import create_unet_model
from ..data import create_dataloader, SegmentationDataset
from ..utils import SegmentationMetrics, LossTracker


class SegmentationTrainer:
    """
    Trainer class for semantic segmentation models
    """
    
    def __init__(self, config_path: str):
        """
        Initialize trainer
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.metrics_calc = None
        self.loss_tracker = LossTracker()
        
        # Setup logging
        self.writer = None
        self.use_wandb = self.config.get('logging', {}).get('use_wandb', False)
        
        # Create directories
        self._create_directories()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _create_directories(self):
        """Create necessary directories"""
        paths = self.config.get('paths', {})
        
        os.makedirs(paths.get('model_save_dir', 'checkpoints'), exist_ok=True)
        os.makedirs(paths.get('log_dir', 'logs'), exist_ok=True)
        os.makedirs(paths.get('output_dir', 'outputs'), exist_ok=True)
    
    def setup_model(self):
        """Setup model, optimizer, scheduler, and loss function"""
        # Create model
        self.model = create_unet_model(self.config)
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        training_config = self.config.get('training', {})
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=training_config.get('learning_rate', 0.001),
            weight_decay=training_config.get('weight_decay', 1e-4)
        )
        
        # Setup scheduler
        scheduler_type = training_config.get('scheduler', 'cosine')
        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=training_config.get('num_epochs', 100)
            )
        elif scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        
        # Setup loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        
        # Setup metrics
        model_config = self.config.get('model', {})
        self.metrics_calc = SegmentationMetrics(
            num_classes=model_config.get('num_classes', 21),
            ignore_index=255
        )
    
    def setup_logging(self):
        """Setup logging (TensorBoard and/or Weights & Biases)"""
        logging_config = self.config.get('logging', {})
        
        # TensorBoard
        if logging_config.get('use_tensorboard', True):
            log_dir = self.config.get('paths', {}).get('log_dir', 'logs')
            self.writer = SummaryWriter(log_dir)
        
        # Weights & Biases
        if self.use_wandb:
            wandb.init(
                project="semantic-segmentation-unet",
                config=self.config,
                name=f"unet_{self.config.get('model', {}).get('num_classes', 21)}classes"
            )
    
    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Create training and validation dataloaders"""
        data_config = self.config.get('data', {})
        training_config = self.config.get('training', {})
        
        # Training dataset
        train_dataset = SegmentationDataset(
            image_dir=data_config.get('train_dir', 'data/train/images'),
            mask_dir=data_config.get('train_dir', 'data/train/masks'),
            image_size=tuple(data_config.get('image_size', [512, 512])),
            augmentation=data_config.get('augmentation'),
            is_training=True
        )
        
        # Validation dataset
        val_dataset = SegmentationDataset(
            image_dir=data_config.get('val_dir', 'data/val/images'),
            mask_dir=data_config.get('val_dir', 'data/val/masks'),
            image_size=tuple(data_config.get('image_size', [512, 512])),
            augmentation=None,
            is_training=False
        )
        
        # Create dataloaders
        train_loader = create_dataloader(
            train_dataset,
            batch_size=training_config.get('batch_size', 8),
            shuffle=True,
            num_workers=data_config.get('num_workers', 4)
        )
        
        val_loader = create_dataloader(
            val_dataset,
            batch_size=training_config.get('batch_size', 8),
            shuffle=False,
            num_workers=data_config.get('num_workers', 4)
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch"""
        self.model.train()
        self.metrics_calc.reset()
        
        total_loss = 0.0
        num_batches = len(train_loader)
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]')
        
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            with torch.no_grad():
                predictions = torch.argmax(outputs, dim=1)
                self.metrics_calc.update(predictions, masks)
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
            })
            
            # Log to TensorBoard
            if self.writer and batch_idx % 10 == 0:
                global_step = epoch * num_batches + batch_idx
                self.writer.add_scalar('Train/Loss', loss.item(), global_step)
        
        avg_loss = total_loss / num_batches
        metrics = self.metrics_calc.compute()
        
        return avg_loss, metrics
    
    def validate_epoch(self, val_loader: DataLoader, epoch: int) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch"""
        self.model.eval()
        self.metrics_calc.reset()
        
        total_loss = 0.0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch+1} [Val]')
            
            for batch_idx, (images, masks) in enumerate(pbar):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                # Update metrics
                predictions = torch.argmax(outputs, dim=1)
                self.metrics_calc.update(predictions, masks)
                
                total_loss += loss.item()
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
                })
        
        avg_loss = total_loss / num_batches
        metrics = self.metrics_calc.compute()
        
        return avg_loss, metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config,
            'loss_tracker': self.loss_tracker
        }
        
        model_save_dir = self.config.get('paths', {}).get('model_save_dir', 'checkpoints')
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(model_save_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(model_save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"New best model saved at epoch {epoch+1}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'loss_tracker' in checkpoint:
            self.loss_tracker = checkpoint['loss_tracker']
        
        return checkpoint['epoch']
    
    def train(self, resume_from: Optional[str] = None):
        """Main training loop"""
        print("Setting up training...")
        
        # Setup components
        self.setup_model()
        self.setup_logging()
        train_loader, val_loader = self.create_dataloaders()
        
        # Resume from checkpoint if specified
        start_epoch = 0
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from)
            print(f"Resumed training from epoch {start_epoch+1}")
        
        training_config = self.config.get('training', {})
        num_epochs = training_config.get('num_epochs', 100)
        early_stopping_patience = training_config.get('early_stopping_patience', 15)
        
        best_metric = 0.0
        patience_counter = 0
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(start_epoch, num_epochs):
            # Training
            train_loss, train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_loss, val_metrics = self.validate_epoch(val_loader, epoch)
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Update loss tracker
            self.loss_tracker.update_train(train_loss, train_metrics)
            self.loss_tracker.update_val(val_loss, val_metrics)
            
            # Logging
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train mIoU: {train_metrics['mean_iou']:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val mIoU: {val_metrics['mean_iou']:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # TensorBoard logging
            if self.writer:
                self.writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
                self.writer.add_scalar('Epoch/Val_Loss', val_loss, epoch)
                self.writer.add_scalar('Epoch/Train_mIoU', train_metrics['mean_iou'], epoch)
                self.writer.add_scalar('Epoch/Val_mIoU', val_metrics['mean_iou'], epoch)
                self.writer.add_scalar('Epoch/Learning_Rate', current_lr, epoch)
            
            # Weights & Biases logging
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_miou': train_metrics['mean_iou'],
                    'val_miou': val_metrics['mean_iou'],
                    'learning_rate': current_lr
                })
            
            # Save checkpoint
            is_best = val_metrics['mean_iou'] > best_metric
            if is_best:
                best_metric = val_metrics['mean_iou']
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Save checkpoint periodically
            save_interval = self.config.get('logging', {}).get('save_interval', 5)
            if (epoch + 1) % save_interval == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        print(f"\nTraining completed! Best validation mIoU: {best_metric:.4f}")
        
        # Close logging
        if self.writer:
            self.writer.close()
        if self.use_wandb:
            wandb.finish()


if __name__ == "__main__":
    # Example usage
    trainer = SegmentationTrainer("config/config.yaml")
    trainer.train()
