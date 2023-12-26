class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = x.mean(dim=1)  # Apply mean pooling across the node dimension
        x = x.mean(dim=1) 
        x = x.mean(dim=0)
               
        x = self.fc(x)
        x = torch.unsqueeze(x, dim=0)
        x = F.softmax(x, dim=1)
        
        return x

