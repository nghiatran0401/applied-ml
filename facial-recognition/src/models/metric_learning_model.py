"""
Metric Learning Face Verification Model
Uses ResNet50 backbone with triplet loss training
"""
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class FaceEmbeddingModel(nn.Module):
    """
    Face embedding model using ResNet50 backbone
    Trained with triplet loss (no classification head)
    """
    def __init__(self, pretrained=True, embedding_dim=512):
        """
        Args:
            pretrained: Use pre-trained ResNet50 weights
            embedding_dim: Dimension of output embedding (can add projection layer)
        """
        super(FaceEmbeddingModel, self).__init__()
        
        # Load pre-trained ResNet50
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Get number of features from the last layer
        num_features = self.backbone.fc.in_features
        
        # Remove classification head
        self.backbone.fc = nn.Identity()
        
        # Optional: Add projection layer to reduce embedding dimension
        if embedding_dim != num_features:
            self.projection = nn.Sequential(
                nn.Linear(num_features, embedding_dim),
                nn.BatchNorm1d(embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, embedding_dim)
            )
        else:
            self.projection = nn.Identity()
        
        self.embedding_dim = embedding_dim
        self.feature_dim = num_features
    
    def forward(self, x):
        """
        Forward pass to extract embeddings
        
        Args:
            x: Input images (batch_size, 3, 224, 224)
            
        Returns:
            embeddings: Face embeddings (batch_size, embedding_dim)
        """
        # Forward through backbone
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Project to embedding dimension
        x = self.projection(x)
        
        # L2 normalize embeddings
        x = F.normalize(x, p=2, dim=1)
        
        return x
    
    def extract_embedding(self, x):
        """
        Extract face embedding (same as forward)
        
        Args:
            x: Input images (batch_size, 3, 224, 224)
            
        Returns:
            embeddings: Face embeddings (batch_size, embedding_dim)
        """
        return self.forward(x)
    
    def freeze_backbone(self, freeze=True):
        """
        Freeze or unfreeze backbone layers
        
        Args:
            freeze: If True, freeze backbone, only train projection
        """
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
        
        # Always keep projection trainable
        if hasattr(self, 'projection'):
            for param in self.projection.parameters():
                param.requires_grad = True


class TripletLoss(nn.Module):
    """
    Triplet Loss for metric learning
    L = max(0, margin + d(anchor, positive) - d(anchor, negative))
    """
    def __init__(self, margin=0.5):
        """
        Args:
            margin: Margin for triplet loss (default 0.5)
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        """
        Compute triplet loss
        
        Args:
            anchor: Anchor embeddings (batch_size, embedding_dim)
            positive: Positive embeddings (batch_size, embedding_dim)
            negative: Negative embeddings (batch_size, embedding_dim)
            
        Returns:
            loss: Triplet loss value
        """
        # Compute distances (using squared Euclidean distance)
        # Since embeddings are L2 normalized, this is equivalent to:
        # d = 2 - 2 * cosine_similarity
        distance_positive = F.pairwise_distance(anchor, positive, p=2)
        distance_negative = F.pairwise_distance(anchor, negative, p=2)
        
        # Triplet loss: max(0, margin + d(a,p) - d(a,n))
        loss = F.relu(self.margin + distance_positive - distance_negative)
        
        return loss.mean()


def create_embedding_model(pretrained=True, embedding_dim=512, freeze_backbone=False):
    """
    Create and configure embedding model for metric learning
    
    Args:
        pretrained: Use pre-trained weights
        embedding_dim: Dimension of output embedding
        freeze_backbone: Freeze backbone layers (fine-tune only projection)
        
    Returns:
        model: Configured model
    """
    model = FaceEmbeddingModel(pretrained=pretrained, embedding_dim=embedding_dim)
    
    if freeze_backbone:
        model.freeze_backbone(freeze=True)
    
    return model
