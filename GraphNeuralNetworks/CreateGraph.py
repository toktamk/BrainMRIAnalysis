from torch_geometric.data import Data
class CreateGraph:
    def __init__(slice_images):
        num_slices = slice_images.shape[0]
        node_features = slice_images
        num_slices, num_features, height, width = node_features.shape
        edge_index = torch.tensor([[i, i+1] for i in range(num_slices-1)], dtype=torch.long).t().contiguous()
        x=torch.tensor(node_features, dtype=torch.float)#.detach().requires_grad_(True)
    
        data = Data(x, edge_index=edge_index)
        x_tensor = torch.tensor(data.x).detach()
        edge_index_tensor = torch.tensor(data.edge_index).detach()

        data_tensor = Data(x=x_tensor, edge_index=edge_index_tensor)
    
        return data_tensor