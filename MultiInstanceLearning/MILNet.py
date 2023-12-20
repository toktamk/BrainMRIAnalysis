import torch
import torch.nn as nn

class MILNet(nn.Module):
    def __init__(self,bagsize):
        super(MILNet, self).__init__()

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
        self.aggregation = nn.MaxPool2d(kernel_size=(bagsize, 1))  # Assuming input bags are of size (4, 4)

        # Bag-level fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(in_features=3136, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=4),  # 2 output classes (positive and negative)
            nn.Softmax()  # Sigmoid activation for binary classification
        )

    def forward(self, bag):
        instance_reps = []
        batch_size,bagsize,C,H,W = bag.size()
        bag = bag.view(batch_size * bagsize, C, H, W)
        
        instance_reps = self.cnn(bag)
        instance_reps = instance_reps.view(batch_size, bagsize, instance_reps.size(1), instance_reps.size(2), 
                                           instance_reps.size(3))
        #Reshape instance-level representations
        bag_rep = torch.mean(instance_reps, dim=1) 
        bag_rep = bag_rep.view(batch_size, -1)
        bag_logits = self.fc(bag_rep)
        
        return bag_logits