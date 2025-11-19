"""
Evaluation script for Metric Learning Face Verification
Evaluates on verification pairs and computes ROC/AUC
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import os
import argparse
from tqdm import tqdm

from src.models.metric_learning_model import create_embedding_model
from src.utils.data_loader import FaceVerificationDataset, get_transforms
from src.utils.evaluation import plot_roc_curve, compute_similarity
from sklearn.metrics import roc_curve, auc, accuracy_score
import numpy as np
import json


def evaluate_metric_learning_model(
    model_path,
    pairs_file,
    data_dir,
    embedding_dim=512,
    save_dir='results',
    device=None
):
    """
    Evaluate metric learning model on verification pairs
    """
    # Device selection: CUDA > CPU
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = create_embedding_model(
        pretrained=False,
        embedding_dim=embedding_dim,
        freeze_backbone=False
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = model.to(device)
    
    print(f"Model loaded. Epoch: {checkpoint.get('epoch', 'N/A')}")
    
    # Create dataset
    print(f"Loading verification pairs from {pairs_file}...")
    dataset = FaceVerificationDataset(
        pairs_file,
        data_dir,
        transform=get_transforms(train=False)
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Evaluating on {len(dataset)} pairs...")
    
    # Extract embeddings and compute similarities
    similarities_cosine = []
    similarities_euclidean = []
    labels = []
    
    with torch.no_grad():
        for img1, img2, label in tqdm(dataloader, desc='Evaluating'):
            img1 = img1.to(device)
            img2 = img2.to(device)
            
            # Extract embeddings
            emb1 = model(img1)
            emb2 = model(img2)
            
            # Compute similarities
            for i in range(len(emb1)):
                e1 = emb1[i].cpu().numpy()
                e2 = emb2[i].cpu().numpy()
                
                # Cosine similarity
                sim_cosine = compute_similarity(e1, e2, metric='cosine')
                sim_euclidean = compute_similarity(e1, e2, metric='euclidean')
                
                similarities_cosine.append(sim_cosine)
                similarities_euclidean.append(sim_euclidean)
                labels.append(label[i].item())
    
    # Convert to numpy arrays
    similarities_cosine = np.array(similarities_cosine)
    similarities_euclidean = np.array(similarities_euclidean)
    labels = np.array(labels)
    
    # Compute ROC curves and AUC
    print("\nEvaluating with Cosine Similarity...")
    fpr_cosine, tpr_cosine, thresholds_cosine = roc_curve(labels, similarities_cosine)
    auc_cosine = auc(fpr_cosine, tpr_cosine)
    
    # Find best threshold (maximizes accuracy)
    accuracies_cosine = []
    for threshold in thresholds_cosine:
        predictions = (similarities_cosine >= threshold).astype(int)
        acc = accuracy_score(labels, predictions)
        accuracies_cosine.append(acc)
    best_idx_cosine = np.argmax(accuracies_cosine)
    best_threshold_cosine = thresholds_cosine[best_idx_cosine]
    best_accuracy_cosine = accuracies_cosine[best_idx_cosine]
    
    results_cosine = {
        'auc': float(auc_cosine),
        'best_threshold': float(best_threshold_cosine),
        'accuracy': float(best_accuracy_cosine),
        'mean_similarity_same': float(similarities_cosine[labels == 1].mean()),
        'mean_similarity_different': float(similarities_cosine[labels == 0].mean())
    }
    
    print("\nEvaluating with Euclidean Distance...")
    fpr_euclidean, tpr_euclidean, thresholds_euclidean = roc_curve(labels, similarities_euclidean)
    auc_euclidean = auc(fpr_euclidean, tpr_euclidean)
    
    # Find best threshold
    accuracies_euclidean = []
    for threshold in thresholds_euclidean:
        predictions = (similarities_euclidean >= threshold).astype(int)
        acc = accuracy_score(labels, predictions)
        accuracies_euclidean.append(acc)
    best_idx_euclidean = np.argmax(accuracies_euclidean)
    best_threshold_euclidean = thresholds_euclidean[best_idx_euclidean]
    best_accuracy_euclidean = accuracies_euclidean[best_idx_euclidean]
    
    results_euclidean = {
        'auc': float(auc_euclidean),
        'best_threshold': float(best_threshold_euclidean),
        'accuracy': float(best_accuracy_euclidean),
        'mean_similarity_same': float(similarities_euclidean[labels == 1].mean()),
        'mean_similarity_different': float(similarities_euclidean[labels == 0].mean())
    }
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot ROC curves
    print("\nPlotting ROC curves...")
    plot_roc_curve(
        fpr_cosine, tpr_cosine, auc_cosine,
        save_path=os.path.join(save_dir, 'metric_learning_roc_cosine.png'),
        title='ROC Curve - Metric Learning (Cosine Similarity)'
    )
    
    plot_roc_curve(
        fpr_euclidean, tpr_euclidean, auc_euclidean,
        save_path=os.path.join(save_dir, 'metric_learning_roc_euclidean.png'),
        title='ROC Curve - Metric Learning (Euclidean Distance)'
    )
    
    # Save results
    results = {
        'model': 'metric_learning',
        'model_path': model_path,
        'cosine_similarity': results_cosine,
        'euclidean_distance': results_euclidean
    }
    
    results_file = os.path.join(save_dir, 'metric_learning_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {results_file}")
    print(f"✓ ROC curves saved to {save_dir}/")
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS - METRIC LEARNING")
    print("=" * 60)
    print(f"\nCosine Similarity:")
    print(f"  AUC: {results_cosine['auc']:.4f}")
    print(f"  Best Threshold: {results_cosine['best_threshold']:.4f}")
    print(f"  Accuracy: {results_cosine['accuracy']:.4f}")
    
    print(f"\nEuclidean Distance:")
    print(f"  AUC: {results_euclidean['auc']:.4f}")
    print(f"  Best Threshold: {results_euclidean['best_threshold']:.4f}")
    print(f"  Accuracy: {results_euclidean['accuracy']:.4f}")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Metric Learning Model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--pairs_file', type=str, required=True,
                       help='Path to verification_pairs_val.txt')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Base directory for verification images')
    parser.add_argument('--embedding_dim', type=int, default=512,
                       help='Embedding dimension (must match model)')
    parser.add_argument('--save_dir', type=str, default='results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    evaluate_metric_learning_model(
        model_path=args.model_path,
        pairs_file=args.pairs_file,
        data_dir=args.data_dir,
        embedding_dim=args.embedding_dim,
        save_dir=args.save_dir
    )

