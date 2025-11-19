import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.classification_model import create_model
from utils.data_loader import create_data_loaders
from utils.evaluation import save_checkpoint

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# learns patterns from training data to predict identities
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Loop through batches (64 images at a time)
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        # Move to GPU
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass (Predicts identities and computes loss
        optimizer.zero_grad()
        outputs = model(images) # Get predictions
        loss = criterion(outputs, labels) # Calculate loss
        
        # Backward pass (Updates model weights)
        loss.backward() # Calculate gradients
        optimizer.step() # Update weights
        
        # Track statistics (Counts correct predictions)
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100*correct/total:.2f}%'
        })
    
    # Calculate average loss and accuracy for the epoch
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

# checks generalization on unseen data (classification accuracy during training)
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad(): # Disable gradient tracking for inference (Saves memory and speed (no backprop)
        for images, labels in tqdm(val_loader, desc='Validating'):
            # Move to GPU
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def train_classification_model(
    classification_data_dir,
    num_epochs=20,
    batch_size=64,
    learning_rate=1e-4,
    save_dir='models',
    device=None
):
    # Device selection: CUDA > CPU
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader, test_loader, num_classes = create_data_loaders(
        classification_data_dir,
        batch_size=batch_size
    )
    
    # Create model
    use_gpu = device == 'cuda'
    model = create_model(
        num_classes=num_classes,
        pretrained=True,
        freeze_backbone=not use_gpu
    )
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Training history
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_val_acc = 0.0
    
    os.makedirs(save_dir, exist_ok=True)
    print(f"\nStarting training for {num_epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 60)
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(save_dir, 'classification_best.pth')
            save_checkpoint(model, optimizer, epoch, val_acc, checkpoint_path)
            print(f"Saved best model (Val Acc: {val_acc:.2f}%)")

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    print(f"Saved training history plot")
    
    # Final test evaluation
    print("\nEvaluating on test set...")
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    # Save final model
    final_checkpoint_path = os.path.join(save_dir, 'classification_final.pth')
    save_checkpoint(model, optimizer, num_epochs-1, test_acc, final_checkpoint_path)
    print(f"Saved final model")
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Classification Model')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to classification_data/ directory')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='models',
                       help='Directory to save models')
    
    args = parser.parse_args()
    
    train_classification_model(
        classification_data_dir=args.data_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_dir=args.save_dir
    )