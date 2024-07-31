#CreateGraph: A module for constructing graphs from image data.

class CreateGraph:
    def __init__(self,api_name=''):
        self.api_name = api_name
        
    def construct_graph(self,images):
        if self.api_name == 'pytorch':
            from torch_geometric.data import Data
            num_slices = images.shape[0]
            node_features = images
            num_slices, num_features, height, width = node_features.shape
            edge_index = torch.tensor([[i, i+1] for i in range(num_slices-1)], dtype=torch.long).t().contiguous()
            x=torch.tensor(node_features, dtype=torch.float)#.detach().requires_grad_(True)
    
            data = Data(x, edge_index=edge_index)
            x_tensor = torch.tensor(data.x).detach()
            edge_index_tensor = torch.tensor(data.edge_index).detach()

            data_tensor = Data(x=x_tensor, edge_index=edge_index_tensor)
    
            return data_tensor
        else:
            import numpy as np
            import networkx as nx
            num_slices = len(images)
            adjacency = np.zeros((num_slices, num_slices))
            graph = nx.Graph()

            for i in range(num_slices):
                graph.add_node(i, features=images[i])

            for i in range(num_slices - 1):
                adjacency[i, i + 1] = 1
                adjacency[i + 1, i] = 1
                graph.add_edge(i, i + 1)

            return graph, adjacency