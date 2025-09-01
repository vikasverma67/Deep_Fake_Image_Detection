import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np
import random
from tqdm import tqdm

# Import custom modules
from src.data.dataset import DeepfakeDetectionDataset, get_data_loaders
from src.data.augmentation import (
    get_train_transforms, 
    get_val_transforms, 
    get_deepfake_specific_transforms,
    AlbumentationsTransform,
    get_albumentations_transforms
)
from src.models.efficientnet import create_model
from src.utils.metrics import DeepfakeMetrics, log_metrics, find_optimal_threshold


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Deepfake Detection Model Training")
    
    # Data related arguments
    parser.add_argument("--data_dir", type=str, default="./src/data/dataset",
                        help="Path to the dataset directory")
    
    # Model related arguments
    parser.add_argument("--model_name", type=str, default="efficientnet-b0",
                        help="Model architecture to use")
    parser.add_argument("--pretrained", action="store_true", default=True,
                        help="Use pretrained weights")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint to resume training")
    
    # Training related arguments
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay for optimizer")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    parser.add_argument("--augmentation", type=str, default="deepfake",
                        choices=["standard", "strong", "deepfake", "albumentations"],
                        help="Augmentation strategy to use")
    parser.add_argument("--img_size", type=int, default=224,
                        help="Image size for training")
    
    # Misc arguments
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--save_dir", type=str, default="./models",
                        help="Directory to save model checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs",
                        help="Directory to save training logs")
    parser.add_argument("--save_freq", type=int, default=1,
                        help="Checkpoint saving frequency (epochs)")
    parser.add_argument("--eval_test", action="store_true", default=False,
                        help="Evaluate on test set after training")
    
    # Parse arguments
    args = parser.parse_args()
    return args


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_epoch(model, train_loader, criterion, optimizer, device, metrics):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    metrics.reset()
    
    # Create progress bar
    pbar = tqdm(train_loader, desc="Training")
    
    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Calculate loss
        loss = criterion(outputs['logits'], labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update metrics
        metrics.update(outputs, labels)
        
        # Update running loss
        running_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': loss.item(),
            'avg_loss': running_loss / (batch_idx + 1)
        })
    
    # Calculate average loss
    avg_loss = running_loss / len(train_loader)
    
    # Compute metrics
    metric_values = metrics.compute()
    metric_values['loss'] = avg_loss
    
    return metric_values


def validate(model, val_loader, criterion, device, metrics, desc="Validation"):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    metrics.reset()
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=desc):
            # Move data to device
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            loss = criterion(outputs['logits'], labels)
            
            # Update metrics
            metrics.update(outputs, labels)
            
            # Update running loss
            running_loss += loss.item()
    
    # Calculate average loss
    avg_loss = running_loss / len(val_loader)
    
    # Compute metrics
    metric_values = metrics.compute()
    metric_values['loss'] = avg_loss
    
    return metric_values


def get_transforms(args):
    """Get transforms based on augmentation strategy"""
    img_size = (args.img_size, args.img_size)
    
    if args.augmentation == "standard":
        train_transform = get_train_transforms(img_size=img_size)
        val_transform = get_val_transforms(img_size=img_size)
    elif args.augmentation == "strong":
        from src.data.augmentation import get_strong_augmentation
        train_transform = get_strong_augmentation(img_size=img_size)
        val_transform = get_val_transforms(img_size=img_size)
    elif args.augmentation == "deepfake":
        train_transform = get_deepfake_specific_transforms(img_size=img_size, is_train=True)
        val_transform = get_deepfake_specific_transforms(img_size=img_size, is_train=False)
    elif args.augmentation == "albumentations":
        train_transform = AlbumentationsTransform(
            get_albumentations_transforms(img_size=img_size, is_train=True)
        )
        val_transform = AlbumentationsTransform(
            get_albumentations_transforms(img_size=img_size, is_train=False)
        )
    else:
        raise ValueError(f"Unknown augmentation strategy: {args.augmentation}")
    
    return train_transform, val_transform


def main():
    """Main training function"""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create directories if they don't exist
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Create TensorBoard writer
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(args.log_dir, f"{args.model_name}_{args.augmentation}_{current_time}")
    writer = SummaryWriter(log_dir=log_dir)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get transforms based on augmentation strategy
    train_transform, val_transform = get_transforms(args)
    print(f"Using {args.augmentation} augmentation strategy")
    
    # Create data loaders
    train_loader, val_loader, test_loader = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=val_transform,
        num_workers=args.num_workers,
        random_seed=args.seed
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    model = create_model(
        model_name=args.model_name,
        pretrained=args.pretrained,
        checkpoint_path=args.checkpoint
    )
    
    # Move model to device
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=3,
        min_lr=1e-6
    )
    
    # Create metrics tracker
    train_metrics = DeepfakeMetrics()
    val_metrics = DeepfakeMetrics()
    test_metrics = DeepfakeMetrics()
    
    # Training loop
    best_val_loss = float('inf')
    best_val_auc = 0.0
    
    print(f"Starting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train for one epoch
        train_metric_values = train_epoch(
            model, train_loader, criterion, optimizer, device, train_metrics
        )
        
        # Log training metrics
        log_metrics(train_metric_values, epoch, prefix="Train", writer=writer)
        
        # Validate the model
        val_metric_values = validate(
            model, val_loader, criterion, device, val_metrics
        )
        
        # Log validation metrics
        log_metrics(val_metric_values, epoch, prefix="Val", writer=writer)
        
        # Update learning rate scheduler
        scheduler.step(val_metric_values['loss'])
        
        # Save best model (by validation loss)
        if val_metric_values['loss'] < best_val_loss:
            best_val_loss = val_metric_values['loss']
            best_model_path = os.path.join(args.save_dir, f"best_model_loss.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
                'metrics': val_metric_values,
                'args': vars(args)
            }, best_model_path)
            print(f"Saved best model (by loss) to {best_model_path}")
        
        # Save best model (by AUC-ROC)
        if val_metric_values['auc_roc'] > best_val_auc:
            best_val_auc = val_metric_values['auc_roc']
            best_model_path = os.path.join(args.save_dir, f"best_model_auc.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'auc_roc': best_val_auc,
                'metrics': val_metric_values,
                'args': vars(args)
            }, best_model_path)
            print(f"Saved best model (by AUC-ROC) to {best_model_path}")
        
        # Save checkpoint periodically
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(args.save_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metric_values,
                'args': vars(args)
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    final_model_path = os.path.join(args.save_dir, "final_model.pth")
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': val_metric_values,
        'args': vars(args)
    }, final_model_path)
    print(f"Saved final model to {final_model_path}")
    
    # Evaluate on test set if requested
    if args.eval_test:
        print("\nEvaluating on test set...")
        model.load_state_dict(torch.load(os.path.join(args.save_dir, "best_model_auc.pth"))['model_state_dict'])
        test_metric_values = validate(
            model, test_loader, criterion, device, test_metrics, desc="Testing"
        )
        log_metrics(test_metric_values, args.epochs, prefix="Test", writer=writer)
        
        # Save test results
        test_results_path = os.path.join(args.save_dir, "test_results.json")
        import json
        with open(test_results_path, 'w') as f:
            json.dump(test_metric_values, f, indent=4)
        print(f"Saved test results to {test_results_path}")
    
    # Close TensorBoard writer
    writer.close()
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main() 