import torch
import torch.nn as nn
from torch.nn import functional as F
class SimCLR(nn.Module):
    def __init__(self, base_encoder, last_channel,projection_dim=128):
        super(SimCLR, self).__init__()
        if base_encoder != None:
            self.base_encoder = base_encoder
            self.last_channel = last_channel
        else:
            self.base_encoder = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace = True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace = True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace = True),
                nn.MaxPool2d(kernel_size=2, stride=2)            
            )
            self.last_channel = 28*28*64
            
        if base_encoder != None:
            self.projection_head = nn.Sequential(
               nn.AdaptiveAvgPool2d(1),  # Global average pooling
                nn.Flatten(),
                nn.Linear(self.last_channel, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, projection_dim))
        else:
            self.projection_head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.last_channel, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, projection_dim))

    def forward(self, x1, x2):
        embedding1 = self.base_encoder(x1)
        embedding2 = self.base_encoder(x2)
        projection1 = self.projection_head(embedding1)
        projection2 = self.projection_head(embedding2)
        return embedding1, embedding2, projection1, projection2

