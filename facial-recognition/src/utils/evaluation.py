import torch
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from pathlib import Path


def compute_similarity(emb1, emb2, metric='cosine'):
    if metric == 'cosine':
        # Cosine similarity: dot product / (norm1 * norm2)
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)
    
    elif metric == 'euclidean':
        # Euclidean distance: sqrt(sum((x1 - x2)^2 + (y1 - y2)^2 + (z1 - z2)^2))
        distance = np.linalg.norm(emb1 - emb2)
        return np.exp(-distance) 

    else:
        raise ValueError(f"Unknown metric: {metric}")


def evaluate_verification_pairs(
    model,
    pairs_file,
    data_dir,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    metric='cosine',
    batch_size=32
):
    model.eval()
    
    # Load pairs
    pairs = []
    with open(pairs_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                img1_path = parts[0]
                img2_path = parts[1]
                label = int(parts[2])
                pairs.append((img1_path, img2_path, label))
    
    print(f"Evaluating {len(pairs)} verification pairs...")
    
    # Transform images to tensor
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    similarities = []
    labels = []
    
    data_dir = Path(data_dir)
    
    with torch.no_grad():
        for img1_path, img2_path, label in tqdm(pairs, desc='Processing pairs'):
            try:
                # Handle paths - remove data_dir prefix if already present
                img1_clean = img1_path
                img2_clean = img2_path
                
                # If path already contains data_dir name, remove it
                if data_dir.name in img1_path:
                    img1_clean = img1_path.replace(data_dir.name + '/', '')
                if data_dir.name in img2_path:
                    img2_clean = img2_path.replace(data_dir.name + '/', '')
                
                # Load images
                img1 = Image.open(data_dir / img1_clean).convert('RGB')
                img2 = Image.open(data_dir / img2_clean).convert('RGB')
                
                # Transform
                img1_tensor = transform(img1).unsqueeze(0).to(device)
                img2_tensor = transform(img2).unsqueeze(0).to(device)
                
                # Extract embeddings
                emb1 = model.extract_embedding(img1_tensor).cpu().numpy()[0]
                emb2 = model.extract_embedding(img2_tensor).cpu().numpy()[0]
                
                # Compute similarity
                sim = compute_similarity(emb1, emb2, metric=metric)
                similarities.append(sim)
                labels.append(label)
                
            except Exception as e:
                print(f"Error processing pair ({img1_path}, {img2_path}): {e}")
                continue
    
    # Compute ROC curve and AUC
    similarities = np.array(similarities)
    labels = np.array(labels)
    
    fpr, tpr, thresholds = roc_curve(labels, similarities)
    auc_score = auc(fpr, tpr)
    
    return similarities, labels, fpr, tpr, thresholds, auc_score


def plot_roc_curve(fpr, tpr, auc_score, save_path=None, title='ROC Curve'):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved ROC curve to {save_path}")
    
    plt.close()


def save_checkpoint(model, optimizer, epoch, accuracy, filepath):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy,
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer=None):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['accuracy']