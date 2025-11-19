"""
Classification model for face verification using ResNet50
"""
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class FaceClassificationModel(nn.Module):

    def __init__(self, num_classes, pretrained=True):
        super(FaceClassificationModel, self).__init__()
        
        self.backbone = models.resnet50(pretrained=pretrained)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)
        self.embedding_dim = num_features
    
    def forward(self, x):
        return self.backbone(x)
    
    def extract_embedding(self, x):
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
        x = F.normalize(x, p=2, dim=1)
        return x
    
    def freeze_backbone(self, freeze=True):
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
        if hasattr(self.backbone, 'fc'):
            for param in self.backbone.fc.parameters():
                param.requires_grad = True


def create_model(num_classes, pretrained=True, freeze_backbone=False):
    model = FaceClassificationModel(num_classes, pretrained=pretrained)
    
    if freeze_backbone:
        model.freeze_backbone(freeze=True)
    
    return model