#MakeDataset: A module for creating a custom dataset for MIL.

from torch.utils.data import Dataset
import torch
class MakeDataset(Dataset):
    def __init__(self, data_bags,target_bags):
        self.input_data = torch.tensor(data_bags).float()
        self.target  = torch.tensor(target_bags).clone().detach()
    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        x = self.input_data[idx]
        y = self.target[idx]
        return x,y