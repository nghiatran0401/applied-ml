"""
Data Loading and Preprocessing for Face Recognition
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path

class FaceClassificationDataset(Dataset):
    """
    Dataset for classification-based face recognition
    Each folder is a person (class)
    """
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir: Directory containing subdirectories, each is a person
            transform: Optional transform to be applied on image
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Get all images and their labels
        self.images = []
        self.labels = []
        self.label_to_id = {}
        self.id_to_label = {}
        
        # Get all person folders
        person_folders = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        
        for label_id, person_folder in enumerate(person_folders):
            person_name = person_folder.name
            self.label_to_id[person_name] = label_id
            self.id_to_label[label_id] = person_name
            
            # Get all images in this folder
            image_files = list(person_folder.glob('*.jpg')) + list(person_folder.glob('*.png'))
            for img_path in image_files:
                self.images.append(str(img_path))
                self.labels.append(label_id)
        
        print(f"Loaded {len(self.images)} images from {len(person_folders)} classes")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_num_classes(self):
        return len(self.label_to_id)


class FaceVerificationDataset(Dataset):
    """
    Dataset for face verification (pair-based)
    """
    def __init__(self, pairs_file, data_dir, transform=None):
        """
        Args:
            pairs_file: Path to verification_pairs_val.txt
            data_dir: Base directory for images
            transform: Optional transform to be applied on image
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Load pairs
        self.pairs = []
        with open(pairs_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    img1_path = parts[0]
                    img2_path = parts[1]
                    label = int(parts[2])  # 1 = same person, 0 = different
                    self.pairs.append((img1_path, img2_path, label))
        
        print(f"Loaded {len(self.pairs)} verification pairs")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        img1_path, img2_path, label = self.pairs[idx]
        
        # Handle paths - remove data_dir prefix if already present
        img1_clean = img1_path
        img2_clean = img2_path
        
        # If path already contains data_dir name, remove it
        if self.data_dir.name in img1_path:
            img1_clean = img1_path.replace(self.data_dir.name + '/', '')
        if self.data_dir.name in img2_path:
            img2_clean = img2_path.replace(self.data_dir.name + '/', '')
        
        # Load images
        img1 = Image.open(self.data_dir / img1_clean).convert('RGB')
        img2 = Image.open(self.data_dir / img2_clean).convert('RGB')
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, label


def get_transforms(train=True, image_size=224):
    """
    Get data augmentation transforms
    
    Args:
        train: If True, apply data augmentation
        image_size: Target image size
        
    Returns:
        transform: torchvision transform
    """
    if train:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])  # ImageNet stats
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    return transform


def create_data_loaders(classification_data_dir, batch_size=32, num_workers=4):
    """
    Create data loaders for classification training
    
    Args:
        classification_data_dir: Directory containing train_data/, val_data/, test_data/
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        
    Returns:
        train_loader, val_loader, test_loader, num_classes
    """
    train_dir = os.path.join(classification_data_dir, 'train_data')
    val_dir = os.path.join(classification_data_dir, 'val_data')
    test_dir = os.path.join(classification_data_dir, 'test_data')
    
    # Create datasets
    train_dataset = FaceClassificationDataset(
        train_dir,
        transform=get_transforms(train=True)
    )
    val_dataset = FaceClassificationDataset(
        val_dir,
        transform=get_transforms(train=False)
    )
    test_dataset = FaceClassificationDataset(
        test_dir,
        transform=get_transforms(train=False)
    )
    
    num_classes = train_dataset.get_num_classes()
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, num_classes

