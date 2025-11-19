import sys
from pathlib import Path
import torch
import argparse
import os
from src.models.classification_model import create_model
from src.utils.evaluation import evaluate_verification_pairs, plot_roc_curve
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Visual flow:
# 1. Load verification pairs from verification_pairs_val.txt
# 2. Read pairs: (face1, face2, label)
# 3. Extract Embeddings
# 4. Compare Embeddings → Similarity Score
# 5. Compare with Threshold → Prediction (Same/Different)
# 6. Compare with Ground Truth → TPR, FPR
# 7. ROC Curve → AUC Score
# 8. Save: JSON + PNG plots

def evaluate_classification_model(
    model_path,
    pairs_file,
    data_dir,
    num_classes,
    save_dir='results',
    device=None
):
    # Device selection: CUDA > CPU
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    model = create_model(num_classes=num_classes, pretrained=False)
    
    # Load model weights
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Evaluate with cosine similarity
    similarities_cosine, labels, fpr_cosine, tpr_cosine, thresholds_cosine, auc_cosine = \
        evaluate_verification_pairs(
            model, pairs_file, data_dir, device=device, metric='cosine'
        )
    print(f"Cosine Similarity - AUC: {auc_cosine:.4f}")
    
    # Evaluate with Euclidean distance
    similarities_euclidean, labels, fpr_euclidean, tpr_euclidean, thresholds_euclidean, auc_euclidean = \
        evaluate_verification_pairs(
            model, pairs_file, data_dir, device=device, metric='euclidean'
        )
    
    print(f"Euclidean Distance - AUC: {auc_euclidean:.4f}")
    
    # Plot ROC curves
    plot_roc_curve(
        fpr_cosine, tpr_cosine, auc_cosine,
        save_path=os.path.join(save_dir, 'roc_curve_cosine.png'),
        title='ROC Curve - Classification Model (Cosine Similarity)'
    )
    
    plot_roc_curve(
        fpr_euclidean, tpr_euclidean, auc_euclidean,
        save_path=os.path.join(save_dir, 'roc_curve_euclidean.png'),
        title='ROC Curve - Classification Model (Euclidean Distance)'
    )
    
    # Save results
    results = {
        'cosine_similarity': {
            'auc': float(auc_cosine),
            'mean_similarity_same': float(similarities_cosine[labels == 1].mean()),
            'mean_similarity_different': float(similarities_cosine[labels == 0].mean()),
        },
        'euclidean_distance': {
            'auc': float(auc_euclidean),
            'mean_similarity_same': float(similarities_euclidean[labels == 1].mean()),
            'mean_similarity_different': float(similarities_euclidean[labels == 0].mean()),
        }
    }
    
    results_path = os.path.join(save_dir, 'classification_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_path}")
    print(f"ROC curves saved to {save_dir}/")
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Cosine Similarity - AUC: {auc_cosine:.4f}")
    print(f"Euclidean Distance - AUC: {auc_euclidean:.4f}")
    print(f"\nBest metric: {'Cosine' if auc_cosine > auc_euclidean else 'Euclidean'}")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Classification Model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--pairs_file', type=str, required=True,
                       help='Path to verification_pairs_val.txt')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Base directory for verification images')
    parser.add_argument('--num_classes', type=int, required=True,
                       help='Number of classes in the model')
    parser.add_argument('--save_dir', type=str, default='results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    evaluate_classification_model(
        model_path=args.model_path,
        pairs_file=args.pairs_file,
        data_dir=args.data_dir,
        num_classes=args.num_classes,
        save_dir=args.save_dir
    )