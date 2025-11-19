"""
Create a subset of the dataset for faster training
Useful when training on CPU or with limited time
"""
import os
import shutil
from pathlib import Path

def create_subset(source_dir, target_dir, num_classes=500, split='train'):
    """
    Create a subset of the dataset
    
    Args:
        source_dir: Source directory (e.g., 'classification_data/train_data')
        target_dir: Target directory (e.g., 'classification_data/train_data_subset')
        num_classes: Number of classes to include
        split: 'train', 'val', or 'test'
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Create target directory
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Get all class folders
    class_folders = sorted([f for f in source_path.iterdir() if f.is_dir()])[:num_classes]
    
    print(f"Creating {split} subset with {len(class_folders)} classes...")
    
    for i, class_folder in enumerate(class_folders):
        if (i + 1) % 50 == 0:
            print(f"  Copied {i + 1}/{len(class_folders)} classes...")
        
        dest_folder = target_path / class_folder.name
        if dest_folder.exists():
            shutil.rmtree(dest_folder)
        shutil.copytree(class_folder, dest_folder)
    
    print(f"✓ Created {split} subset: {len(class_folders)} classes")
    return len(class_folders)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create dataset subset')
    parser.add_argument('--num_classes', type=int, default=500,
                       help='Number of classes to include')
    parser.add_argument('--source', type=str, required=True,
                       help='Source directory (e.g., classification_data/train_data)')
    parser.add_argument('--target', type=str, required=True,
                       help='Target directory (e.g., classification_data/train_data_subset)')
    
    args = parser.parse_args()
    
    create_subset(args.source, args.target, args.num_classes)

