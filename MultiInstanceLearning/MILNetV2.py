import torch
import torch.nn as nn
import torch.nn.functional as F
class MILNetV2(nn.Module):
    def __init__(self,bagsize):
        super(MILNetV2, self).__init__()

        # Instance-level CNN layers
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Aggregation layer
        #self.aggregation = MaxPool2d(kernel_size=(bagsize, 1))  

        # Bag-level fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(in_features=3136, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(inplace=True),
            
        )
        self.output_layer = nn.Linear(in_features=256*bagsize, out_features=4) 

    def forward(self, bag):
        instance_reps = []
        batch_size,bagsize,C,H,W = bag.size()
        bag = bag.view(batch_size * bagsize, C, H, W)
        bag_rep = self.cnn(bag)
        bag_rep = bag_rep.view(batch_size * bagsize,-1)
        
        #instance_reps = instance_reps.view(batch_size, bagsize,instance_reps.size(1),instance_reps.size(2),instance_reps.size(3))
        #Reshape instance-level representations
        #bag_rep = torch.mean(instance_reps, dim=1) 
        #bag_rep = bag_rep.view(batch_size, -1)
        bag_rep = self.fc(bag_rep)
        instance_rep = bag_rep.view(batch_size,-1)
        instance_rep = self.output_layer(instance_rep)
        output = F.softmax(instance_rep, dim=1)
        
        return output