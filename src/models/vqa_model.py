import torch.nn as nn

class VQAModel(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(VQAModel, self).__init__()
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim + 768, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.classifier = nn.Linear(512, num_classes)
    
    def forward(self, img_features, text_features):
        combined = torch.cat([img_features, text_features], dim=1)
        fused = self.fusion(combined)
        return self.classifier(fused)
