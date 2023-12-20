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
import ContrastiveLossClass

import torchvision.models as models

# Step 2: Data Loader
dataset = MRIData.MRIData(root='')
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
def encoder(encoderName):
    if encoderName == None:
        base_encoder = None
        last_channel = 0
    else:
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
    return (base_encoder,last_channel)

encoderName = None
base_encoder, last_channel = encoder(encoderName)

# Step 4: Projection Head
if encoderName != None: 
    projection_head = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),  # Global average pooling
        nn.Flatten(),
        nn.Linear(last_channel, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 128)
    )
else:
        projection_head = nn.Sequential(
        nn.Flatten(),
        nn.Linear(last_channel, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 128)
    )

# Step 6: Training Loop
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimCLR.SimCLR(base_encoder,last_channel)
#model = model.to(device)
criterion = ContrastiveLossClass.ContrastiveLoss()
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