import torch
import torch.nn as nn

class RewardModel(nn.Module):
    def __init__(self, feature_extractor, mlp_head):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.mlp_head = mlp_head
        
    def forward(self, text_embeddings, images):
        # Extract features
        image_features = self.feature_extractor(images)
        text_features = self.feature_extractor(text_embeddings)
        
        # Cross attention
        combined_features = self.cross_attention(image_features, text_features)
        
        # Generate reward score
        reward = self.mlp_head(combined_features)
        return reward 