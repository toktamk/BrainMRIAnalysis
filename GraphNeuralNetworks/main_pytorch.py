#This project implements a Graph Neural Network (GNN) for classifying MRI images using PyTorch and PyTorch Geometric. The workflow involves reading MRI data, generating bags of images, constructing graphs, and training a GNN model to classify the MRI scans.
#This project consists of several key components:
#MRIDataRead: A module for reading and preprocessing MRI data.
#CreateGraph: A module for constructing graphs from image data.
#GraphDataset: A custom dataset class for handling graph data.
#Train_Evaluate_Model: A module for training and evaluating the GNN model.

from torch_geometric.data import Data
from torch.utils.data.dataloader import default_collate

def custom_collate(batch):
    elem = batch[0]
    if isinstance(elem, Data):
        return batch
    elif isinstance(elem, tuple):
        return tuple(custom_collate(samples) for samples in zip(*batch))
    elif isinstance(elem, list):
        transposed = zip(*batch)
        return [custom_collate(samples) for samples in transposed]
    else:
        return default_collate(batch)

import torch
import torch.nn as nn
import GraphDataset
import CreateGraph
import torch.optim as optim
from torch.nn import functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool  

from torch.utils.data import Dataset
import torch
import MRIDataRead
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import MRIDataRead
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import Train_Evaluate_Model
root = "D://datasets//MRI_Mahdieh_Datasets//task1//"
newdim = 224
mri_data_read = MRIDataRead.MRIDataRead(root,newdim)
total_mris,targets,channel_num = mri_data_read.ReadData()
bagsize = 5
data_bags, target_bags = mri_data_read.GenBags(total_mris,targets,bagsize)
data_bags = np.array(data_bags)
data_bags = data_bags.astype(float)
target_bags = np.array(target_bags)
target_bags = torch.from_numpy(target_bags).long()
target = F.one_hot(target_bags)
target_bags = target[:, 1:]
train_data, test_data, train_labels, test_labels = train_test_split(data_bags, target_bags, test_size=0.2)

train_list = []
test_list = []

for i in range(train_data.shape[0]):
    slice_images = torch.tensor(train_data[i,:,:])
    slice_images = slice_images.reshape(-1,newdim,newdim,channel_num)
    
    data= CreateGraph(slice_images)
    train_list.append(data)
    

    
for i in range(test_data.shape[0]):
    slice_images = torch.tensor(test_data[i,:,:])
    slice_images = slice_images.reshape(-1,newdim,newdim,channel_num)
    data = CreateGraph(slice_images)
    test_list.append(data)
    

train_dataset = GraphDataset(train_list,train_labels)
test_dataset = GraphDataset(test_list,test_labels)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,collate_fn=custom_collate)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,collate_fn=custom_collate)

# Set device (CPU or GPU)
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device = 'cpu'

input_dim = channel_num # Dimensionality of the input node features
hidden_dim = 128  # Dimensionality of the hidden layers
output_dim = train_labels.shape[1] # Dimensionality of the output (number of classes)


model = GNNModel(input_dim, hidden_dim, output_dim)#.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
Train_Evaluate_Model = Train_Evaluate_Model(model,device,optimizer,criterion)
num_epochs = 50
for epoch in range(num_epochs):
    Train_Evaluate_Model.train(model, device,train_loader, optimizer, criterion)#, device)
    train_acc = Train_Evaluate_Model.evaluate(model, device, train_loader)#,device)
    test_acc = Train_Evaluate_Model.evaluate(model,device, test_loader)#, device)
    print(f'Epoch: {epoch+1}/{num_epochs}, Train Accuracy: {train_acc:.2f}%, Test Accuracy: {test_acc:.2f}%')

# Reset the warning filter to its default behavior (optional)
warnings.resetwarnings()