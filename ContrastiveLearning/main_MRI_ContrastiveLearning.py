#This code sets up the data loader for the MRI dataset and prepares the encoder for use in the subsequent steps of the SimCLR training process.
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

# Step 1: Data Loader
#The MRIData class is used to create a dataset object for the MRI data.
#The root parameter specifies the directory where the MRI data is stored
dataset = MRIData.MRIData(root='') 
#The DataLoader class is used to create a data loader object 
#that will handle the loading and batching of the MRI data.
#The dataset parameter specifies the dataset object to be used.
#batch_size=64 sets the number of samples in each batch to 64.
#shuffle=True shuffles the data samples before each epoch to introduce randomness and prevent overfitting.
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
#The encoder function takes an encoderName parameter and returns a base encoder and the number of channels in the last layer of the encoder.
def encoder(encoderName):
    if encoderName == None:
        base_encoder = None
        last_channel = 0
    else:
        #it loads the pre-trained mobilenet_v2 model 
        base_encoder = models.mobilenet_v2(weights=True)
        # The last layer of the base encoder is removed 
        base_encoder = torch.nn.Sequential(*list(base_encoder.children())[:-1])
        # Freeze the parameters of the base_encoder
        for param in base_encoder.parameters():
            param.requires_grad = False
        # The parameters of the base encoder are frozen
        for param in base_encoder.parameters():
            param.requires_grad = False
        #The number of channels in the last layer is stored in last_channel.
        last_channel = base_encoder[-1][-1].out_channels
    return (base_encoder,last_channel)

#Step 2: encoder model
encoderName = None
base_encoder, last_channel = encoder(encoderName)

# Step 3: Projection Head
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

# Step 4: Training Loop
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimCLR.SimCLR(base_encoder,last_channel)
#model = model.to(device)
criterion = ContrastiveLossClass.ContrastiveLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = CosineAnnealingLR(optimizer, T_max=len(dataloader), eta_min=0)
epochs_num = 100
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