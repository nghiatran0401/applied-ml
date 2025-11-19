"""
Classification-Based Face Verification Model
ResNet50 is a CNN architecture designed to excel in image classification and other CV tasks. It belongs to the ResNet family, introduced by Microsoft Research in 2015 to address the difficulty of training DL networks due to the vanishing gradient problem. 
ResNet50 is built with 50 layers that process the image step by step, finding simple patterns like edges first, then more complex shapes, and finally understanding the whole picture. What makes ResNet50 special is its shortcuts. These let information jump over some layers instead of going through all of them, kind of like taking a shortcut on a path. This helps the computer learn faster and better, especially when the network gets very deep with many layers.
"""
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class FaceClassificationModel(nn.Module):

    def __init__(self, num_classes, pretrained=True):
        super(FaceClassificationModel, self).__init__()
        
        # Load pre-trained ResNet50
        # Uses weights learned from millions of images
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Get number of features from the last layer (typically 2048)
        num_features = self.backbone.fc.in_features
        
        # Replace final layer for face classification
        # Original: outputs 1000 classes (ImageNet)
        # Modified: outputs num_classes (e.g., 4000 face identities)
        self.backbone.fc = nn.Linear(num_features, num_classes)
        
        # Store num_features for embedding extraction
        self.embedding_dim = num_features
    
    # Forward pass: used for training (predicting identities) and evaluation (extracting embeddings)
    # Input: 64 face images
    # Output: 64 x 4000 matrix (scores per identity)
    def forward(self, x):
        return self.backbone(x)
    
    # Used for: face verification (comparing two faces using cosine/Euclidean distance)
    # Input: 64 face images
    # Output: 64 x 2048 matrix (features per image)
    def extract_embedding(self, x):
        # Forward through backbone (without final layer)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        # Process through each layer (1-4)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # Average pooling: reduces spatial dimensions to 1x1
        x = self.backbone.avgpool(x)

        # Flatten: convert 1x1x2048 to 2048 (features before classification)
        x = torch.flatten(x, 1)
        
        # L2 normalize embeddings
        # Normalize to unit length (Euclidean distance)
        x = F.normalize(x, p=2, dim=1)
        
        return x
    
    # Freeze or unfreeze backbone layers (for training optimization)
    # When to use:
    # - CPU or limited time: freeze backbone
    # - GPU and time available: unfreeze for better accuracy
    def freeze_backbone(self, freeze=True):
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
        
        # Always keep final layer trainable
        if hasattr(self.backbone, 'fc'):
            for param in self.backbone.fc.parameters():
                param.requires_grad = True


def create_model(num_classes, pretrained=True, freeze_backbone=False):
    model = FaceClassificationModel(num_classes, pretrained=pretrained)
    
    if freeze_backbone:
        model.freeze_backbone(freeze=True)
    
    return model