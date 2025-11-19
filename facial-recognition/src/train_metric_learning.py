"""
Training script for Metric Learning Face Verification
Uses triplet loss with hard negative mining
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.models.metric_learning_model import create_embedding_model, TripletLoss
from src.utils.data_loader import FaceClassificationDataset, get_transforms
from src.utils.triplet_mining import batch_hard_triplet_mining
from src.utils.evaluation import save_checkpoint, load_checkpoint


def train_epoch(model, train_loader, criterion, optimizer, device, margin=0.5):
    """Train for one epoch with triplet loss"""
    model.train()
    running_loss = 0.0
    num_triplets = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Extract embeddings
        embeddings = model(images)
        
        # Batch hard triplet mining
        anchor, positive, negative = batch_hard_triplet_mining(
            embeddings, labels, margin=margin
        )
        
        if anchor is None:
            continue  # No valid triplets in this batch
        
        # Forward pass
        optimizer.zero_grad()
        loss = criterion(anchor, positive, negative)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        num_triplets += len(anchor)
        
        pbar.set_postfix({'loss': loss.item(), 'triplets': len(anchor)})
    
    epoch_loss = running_loss / max(num_triplets, 1)
    return epoch_loss


def validate(model, val_loader, criterion, device, margin=0.5):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    num_triplets = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validating'):
            images = images.to(device)
            labels = labels.to(device)
            
            # Extract embeddings
            embeddings = model(images)
            
            # Batch hard triplet mining
            anchor, positive, negative = batch_hard_triplet_mining(
                embeddings, labels, margin=margin
            )
            
            if anchor is None:
                continue
            
            # Compute loss
            loss = criterion(anchor, positive, negative)
            
            running_loss += loss.item()
            num_triplets += len(anchor)
    
    epoch_loss = running_loss / max(num_triplets, 1)
    return epoch_loss


def train_metric_learning_model(
    classification_data_dir,
    num_epochs=20,
    batch_size=64,
    learning_rate=1e-4,
    margin=0.5,
    embedding_dim=512,
    save_dir='models',
    device=None
):
    """
    Train metric learning model with triplet loss
    
    Args:
        classification_data_dir: Path to classification_data/ directory
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        margin: Margin for triplet loss
        embedding_dim: Dimension of face embeddings
        save_dir: Directory to save model
        device: 'cuda' or 'cpu' (auto-detected if None)
    """
    # Device selection: CUDA > CPU
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    
    print(f"Using device: {device}")
    
    # Create data loaders
    print("Loading data...")
    train_dir = os.path.join(classification_data_dir, 'train_data')
    val_dir = os.path.join(classification_data_dir, 'val_data')
    
    train_dataset = FaceClassificationDataset(
        train_dir,
        transform=get_transforms(train=True)
    )
    val_dataset = FaceClassificationDataset(
        val_dir,
        transform=get_transforms(train=False)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create model
    print("Creating model...")
    # With GPU, we can train more layers for better accuracy
    use_gpu = device == 'cuda'
    model = create_embedding_model(
        pretrained=True,
        embedding_dim=embedding_dim,
        freeze_backbone=not use_gpu  # Unfreeze if GPU available
    )
    model = model.to(device)
    
    # Loss and optimizer
    criterion = TripletLoss(margin=margin)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Training history
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 60)
        
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, margin=margin
        )
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device, margin=margin)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(save_dir, 'metric_learning_best.pth')
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
            print(f"✓ Saved best model (Val Loss: {val_loss:.4f})")
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
    
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Triplet Loss')
    plt.legend()
    plt.title('Training and Validation Loss (Metric Learning)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metric_learning_training_history.png'))
    print(f"\n✓ Saved training history plot")
    
    # Save final model
    final_checkpoint_path = os.path.join(save_dir, 'metric_learning_final.pth')
    save_checkpoint(model, optimizer, num_epochs-1, val_loss, final_checkpoint_path)
    print(f"✓ Saved final model")
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print("=" * 60)
    
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Metric Learning Model')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to classification_data/ directory')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--margin', type=float, default=0.5,
                       help='Margin for triplet loss')
    parser.add_argument('--embedding_dim', type=int, default=512,
                       help='Embedding dimension')
    parser.add_argument('--save_dir', type=str, default='models',
                       help='Directory to save models')
    
    args = parser.parse_args()
    
    train_metric_learning_model(
        classification_data_dir=args.data_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        margin=args.margin,
        embedding_dim=args.embedding_dim,
        save_dir=args.save_dir
    )

