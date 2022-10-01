import torch
import torch.nn as nn

from torch_geometric.nn import SAGEConv, GATConv, Linear, to_hetero

import copy

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_channels, add_self_loops=False)
        self.lin1 = Linear(-1, hidden_channels)
        self.conv2 = GATConv((-1, -1), out_channels, add_self_loops=False)
        self.lin2 = Linear(-1, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index) + self.lin1(x)
        x = x.relu()
        x = self.conv2(x, edge_index) + self.lin2(x)
        return x

class SAGE(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class ConvGR(torch.nn.Module):
    def __init__(self, hidden_channels, metadata, num_node_info, gnn_type, device):
        super().__init__()
        self.device = device
        if gnn_type == "SAGE":
            self.gnn = SAGE(hidden_channels, num_node_info['item'])
        elif gnn_type == "GAT":
            self.gnn = GAT(hidden_channels, num_node_info['item'])
        else:
            print("No such type")
            exit()        
        self.gnn = to_hetero(self.gnn, metadata, aggr='sum')

        self.emb = {}
        for key in num_node_info:
            self.emb[key] = nn.Embedding(num_node_info[key], hidden_channels).to(device)

    def forward(self, x, edge_index):
        x = copy.copy(x)
        for node_type in self.emb:
            x[node_type] = self.emb[node_type](x[node_type])

        x = self.gnn(x, edge_index)


        return x

