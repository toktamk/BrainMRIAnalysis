import torch
import torch.nn as nn
from torch.nn import functional as F
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, embedding1, embedding2, projection1, projection2):
        batch_size = embedding1.shape[0]
        embeddings = torch.cat([embedding1, embedding2], dim=0)
        projections = torch.cat([projection1, projection2], dim=0)

        similarity_matrix = F.cosine_similarity(projections.unsqueeze(1), projections.unsqueeze(0), dim=2)
        mask = torch.eye(batch_size * 2, device=embeddings.device, dtype=torch.bool)
        similarity_matrix = similarity_matrix.masked_fill(mask, -1e9)

        logits = similarity_matrix / self.temperature
        labels = torch.arange(batch_size * 2, device=embeddings.device)
        loss = F.cross_entropy(logits, labels)
        return loss