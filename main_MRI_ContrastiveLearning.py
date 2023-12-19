import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import MRIData
import cv2
import numpy as np
import SimCLR
# Step 3: Base Encoder
import torchvision.models as models
# Step 1: Data Augmentation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# Step 2: Data Loader
dataset = MRIData.MRIData(root='')#, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
base_encoder = models.mobilenet_v2(weights=True)
# Remove the last layer
base_encoder = torch.nn.Sequential(*list(base_encoder.children())[:-1])

# Freeze the parameters of the base_encoder
for param in base_encoder.parameters():
    param.requires_grad = False
# Freeze all the pre-trained parameters
for param in base_encoder.parameters():
    param.requires_grad = False
last_channel = base_encoder[-1][-1].out_channels
# Step 4: Projection Head
projection_head = nn.Sequential(
    nn.AdaptiveAvgPool2d(1),  # Global average pooling
    nn.Flatten(),
    nn.Linear(last_channel, 128),
    nn.ReLU(inplace=True),
    nn.Linear(128, 128),
)

# Step 5: Contrastive Loss
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

# Step 6: Training Loop
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimCLR.SimCLR(base_encoder,last_channel)
#model = model.to(device)
criterion = ContrastiveLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = CosineAnnealingLR(optimizer, T_max=len(dataloader), eta_min=0)
epochs_num = 2
for epoch in range(epochs_num):
    model.train()
    total_loss = 0.0
    for image_pair in dataloader:
        image_pair = np.array(image_pair) 
        image_pair = image_pair.reshape(-1,2,1,224,224)
        tmp1 = np.repeat(image_pair[:, 0],3,axis=1)
        tmp2 = np.repeat(image_pair[:, 1],3,axis=1)
        tmp1 = torch.from_numpy(tmp1).float()
        tmp2 = torch.from_numpy(tmp2).float()
        optimizer.zero_grad()
        embedding1, embedding2, projection1, projection2 = model(tmp1,tmp2)
        loss = criterion(embedding1, embedding2, projection1, projection2)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    print(f"Epoch {epoch+1} Loss: {total_loss / len(dataloader)}")