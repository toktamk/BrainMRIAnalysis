from torch.utils.data import Dataset

class GraphDataset(Dataset):
    def __init__(self, data_list,labels):
        
              
        self.data_list = data_list
        self.labels = labels

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]
        label = self.labels[index]
        #print(data)
        return data, label

