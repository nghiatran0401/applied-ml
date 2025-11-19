"""
Triplet Mining Utilities
Hard negative mining for better triplet selection
"""
import torch
import numpy as np
from collections import defaultdict


def get_triplets(embeddings, labels, margin=0.5):
    """
    Generate triplets from embeddings and labels
    Uses hard negative mining: selects hardest negative for each anchor-positive pair
    """
    embeddings = embeddings.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    
    # Group embeddings by label
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)
    
    anchor_indices = []
    positive_indices = []
    negative_indices = []
    
    # For each label, create triplets
    for label, indices in label_to_indices.items():
        if len(indices) < 2:
            continue  # Need at least 2 samples for anchor-positive pair
        
        # Get embeddings for this label
        label_embeddings = embeddings[indices]
        
        # Compute pairwise distances within this label
        # (for hard positive mining - select farthest positive)
        if len(indices) > 2:
            pairwise_distances = np.sqrt(
                np.sum((label_embeddings[:, None, :] - label_embeddings[None, :, :]) ** 2, axis=2)
            )
            # Get hardest positive (farthest pair)
            max_dist_idx = np.unravel_index(np.argmax(pairwise_distances), pairwise_distances.shape)
            anchor_idx = indices[max_dist_idx[0]]
            positive_idx = indices[max_dist_idx[1]]
        else:
            anchor_idx = indices[0]
            positive_idx = indices[1]
        
        anchor_embedding = embeddings[anchor_idx]
        
        # Find hardest negative (closest negative sample)
        min_negative_distance = float('inf')
        best_negative_idx = None
        
        for other_label, other_indices in label_to_indices.items():
            if other_label == label:
                continue
            
            # Check all samples from other labels
            for neg_idx in other_indices:
                negative_embedding = embeddings[neg_idx]
                distance = np.linalg.norm(anchor_embedding - negative_embedding)
                
                # Check if this is a valid hard negative
                # (distance should be less than positive distance + margin)
                positive_distance = np.linalg.norm(anchor_embedding - embeddings[positive_idx])
                
                if distance < positive_distance + margin and distance < min_negative_distance:
                    min_negative_distance = distance
                    best_negative_idx = neg_idx
        
        if best_negative_idx is not None:
            anchor_indices.append(anchor_idx)
            positive_indices.append(positive_idx)
            negative_indices.append(best_negative_idx)
    
    return anchor_indices, positive_indices, negative_indices


def batch_hard_triplet_mining(embeddings, labels, margin=0.5):
    """
    Batch hard triplet mining
    For each anchor, find hardest positive and hardest negative in the batch
    """
    # Compute pairwise distances (since embeddings are normalized, use cosine distance)
    # distance = 1 - cosine_similarity = 1 - (a^T * b)
    pairwise_distances = torch.cdist(embeddings, embeddings, p=2)  # Euclidean distance
    
    # Create mask for same label and different label
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)  # (batch, batch)
    labels_different = ~labels_equal
    
    # Mask diagonal (same sample)
    mask_same = labels_equal.clone()
    mask_same.fill_diagonal_(False)
    
    anchor_indices = []
    positive_indices = []
    negative_indices = []
    
    for i in range(len(embeddings)):
        anchor_idx = i
        
        # Find hardest positive (farthest same-label sample)
        positive_mask = mask_same[i]
        if positive_mask.any():
            positive_distances = pairwise_distances[i][positive_mask]
            hardest_positive_idx = torch.where(positive_mask)[0][torch.argmax(positive_distances)]
            positive_idx = hardest_positive_idx.item()
        else:
            continue  # No positive sample in batch
        
        # Find hardest negative (closest different-label sample)
        negative_mask = labels_different[i]
        if negative_mask.any():
            negative_distances = pairwise_distances[i][negative_mask]
            hardest_negative_idx = torch.where(negative_mask)[0][torch.argmin(negative_distances)]
            negative_idx = hardest_negative_idx.item()
        else:
            continue  # No negative sample in batch
        
        # Check if triplet is valid (positive distance < negative distance)
        positive_distance = pairwise_distances[i, positive_idx]
        negative_distance = pairwise_distances[i, negative_idx]
        
        if positive_distance < negative_distance + margin:
            anchor_indices.append(anchor_idx)
            positive_indices.append(positive_idx)
            negative_indices.append(negative_idx)
    
    if len(anchor_indices) == 0:
        return None, None, None
    
    anchor = embeddings[anchor_indices]
    positive = embeddings[positive_indices]
    negative = embeddings[negative_indices]
    
    return anchor, positive, negative


def create_triplet_dataset(dataset, num_triplets_per_epoch=None):
    """
    Create triplet dataset from classification dataset
        List of (anchor_idx, positive_idx, negative_idx) tuples
    """
    # Group images by label
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(dataset.labels):
        label_to_indices[label].append(idx)
    
    triplets = []
    labels_list = list(label_to_indices.keys())
    
    # Generate triplets
    for anchor_label in labels_list:
        anchor_indices = label_to_indices[anchor_label]
        
        if len(anchor_indices) < 2:
            continue  # Need at least 2 samples for anchor-positive
        
        # Get negative labels (different from anchor)
        negative_labels = [l for l in labels_list if l != anchor_label]
        
        if len(negative_labels) == 0:
            continue
        
        # Create triplets for this anchor label
        for anchor_idx in anchor_indices:
            # Select positive (different sample from same label)
            positive_candidates = [idx for idx in anchor_indices if idx != anchor_idx]
            if len(positive_candidates) == 0:
                continue
            
            positive_idx = np.random.choice(positive_candidates)
            
            # Select negative (random sample from different label)
            negative_label = np.random.choice(negative_labels)
            negative_idx = np.random.choice(label_to_indices[negative_label])
            
            triplets.append((anchor_idx, positive_idx, negative_idx))
    
    if num_triplets_per_epoch and len(triplets) > num_triplets_per_epoch:
        # Randomly sample triplets
        triplets = np.random.choice(len(triplets), num_triplets_per_epoch, replace=False)
        triplets = [triplets[i] for i in triplets]
    
    return triplets

