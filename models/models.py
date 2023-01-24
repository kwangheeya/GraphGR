import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, GATv2Conv, Linear, AGNNConv
from torch_geometric.nn import to_hetero, to_hetero_with_bases, MessagePassing
import numpy as np

import copy


class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, drop_rate=0.0):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_channels, add_self_loops=False)
        self.lin1 = Linear(-1, hidden_channels)
        self.conv2 = GATConv((-1, -1), out_channels, add_self_loops=False)
        self.lin2 = Linear(-1, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index) + self.lin1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index) + self.lin2(x)
        x = F.relu(x)
        return x

class GATv2(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, drop_rate=0.0):
        super().__init__()
        self.conv1 = GATv2Conv((-1, -1), hidden_channels, add_self_loops=False)
        self.lin1 = Linear(-1, hidden_channels)
        self.conv2 = GATv2Conv((-1, -1), out_channels, add_self_loops=False)
        self.lin2 = Linear(-1, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index) + self.lin1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index) + self.lin2(x)
        x = F.relu(x)
        return x

class SAGE(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, drop_rate=0.2):
        super().__init__()
        self.conv1 = SAGEConv(-1, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.drop_rate = drop_rate
        

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index) 
        x = F.relu(x)
        x = self.conv2(x, edge_index) 
        x = F.relu(x)
        return x


class GraphGR(torch.nn.Module):
    def __init__(self, hidden_channels, metadata, num_node_info, gnn_type, device, drop_rate=0.2, edge_drop=0.95):
        super().__init__()
        self.device = device
        self.drop_rate = drop_rate
        self.edge_drop = edge_drop

        # Embedding
        self.emb = {}
        for key in num_node_info:
            self.emb[key] = nn.Embedding(num_node_info[key], hidden_channels).to(device)

        # Randomized prior
        self.randgnn = SAGEConv(-1, hidden_channels)

        # GNN for inductive learning
        if gnn_type == 'GAT':
            gnn = GAT(hidden_channels, hidden_channels, drop_rate=drop_rate)
        elif gnn_type == 'GATv2':
            gnn = GATv2(hidden_channels, hidden_channels, drop_rate=drop_rate)
        elif gnn_type == 'SAGE':
            gnn = SAGE(hidden_channels, hidden_channels, drop_rate=drop_rate)
        self.gnn = to_hetero(gnn, metadata, aggr='sum')    


        # Group rating predictor
        self.predictor = nn.Sequential(
            #Linear(-1, hidden_channels*2),
            #nn.ReLU(),
            #nn.Dropout(p=drop_rate),
            Linear(-1, num_node_info['item']),
        )

        # User rating predictor
        self.predictor_user = nn.Sequential(
            #Linear(-1, hidden_channels*2),
            #nn.ReLU(),
            #nn.Dropout(p=drop_rate),
            Linear(-1, num_node_info['item']),
        )
        # rating predictor for teacher
        self.predictor_tea = nn.Sequential(
            #Linear(-1, hidden_channels*2),
            #nn.ReLU(),
            #nn.Dropout(p=drop_rate),
            Linear(-1, num_node_info['item']),
        )
        self.mseloss = nn.MSELoss()


    def forward(self, x, edge_index):       
        emb_x = copy.copy(x)

        for node_type in self.emb:                
            emb_x[node_type] = self.emb[node_type](x[node_type]) 
            if node_type == 'group':
                emb_x[node_type] = emb_x[node_type] * 0 
            else:
                emb_x[node_type] = F.dropout(emb_x[node_type], self.drop_rate, training=self.training)

        if self.training:     
            # teacher model
            rep_tea = self.gnn(emb_x, edge_index)
            out_tea = {}        
            rep_tea['group'] = F.dropout(rep_tea['group'], self.drop_rate)
            out_tea['group'] = self.predictor_tea(rep_tea['group'])               
           
            # Random augmentation
            edge_index_aug = copy.copy(edge_index)

            edge_n = edge_index[('group', '', 'item')].shape[1]
            sampled_int_arr = np.random.permutation(edge_n)[:int(edge_n*(1.0-self.edge_drop))]
            sampled_edge_index = edge_index[('group', '', 'item')][:,sampled_int_arr]

            edge_index_aug[('group', '', 'item')] = sampled_edge_index
            edge_index_aug[('item', 'rev_', 'group')] = torch.stack((sampled_edge_index[1], sampled_edge_index[0]))

            rep_aug = self.gnn(emb_x, edge_index_aug)
            out_aug = {}
            rep_aug['group'] = F.dropout(rep_aug['group'], self.drop_rate)
            out_aug['group'] = self.predictor(rep_aug['group']) 

            rep_aug['user'] = F.dropout(rep_aug['user'], self.drop_rate)
            out_aug['user'] = self.predictor_user(rep_aug['user'])

            # KL-divergence Loss for knowledge distillation
            tea_dist = F.softmax(out_tea['group'], dim=-1)
            aug_dist = F.log_softmax(out_aug['group'], dim=-1)
            kd_loss = F.kl_div(aug_dist, tea_dist, reduction='batchmean')

            # Neighbor knowledge distillation
            group_reps = out_aug['group'][edge_index[('group', '', 'user')][0]]
            user_reps = out_aug['user'][edge_index[('group', '', 'user')][1]]

            #group_dist = F.log_softmax(group_reps, dim=-1)
            #user_dist = F.softmax(user_reps, dim=-1)
            #kd_loss += F.kl_div(group_dist, user_dist, reduction='batchmean')   
            group_dist = F.softmax(group_reps, dim=-1)
            user_dist = F.log_softmax(user_reps, dim=-1)
            kd_loss2 = F.kl_div(user_dist, group_dist, reduction='batchmean')  
            return out_tea, out_aug, kd_loss, kd_loss2
        else:
            rep = self.gnn(emb_x, edge_index)
            out = {}
            out['group'] = self.predictor(rep['group'])             
            return out




class BaseGR(torch.nn.Module):
    def __init__(self, hidden_channels, metadata, num_node_info, gnn_type, device, drop_rate=0.2):
        super().__init__()
        self.device = device
        self.drop_rate = drop_rate

        # Embedding
        self.emb = {}
        for key in num_node_info:
            self.emb[key] = nn.Embedding(num_node_info[key], hidden_channels).to(device)

        # GNN for inductive learning
        gnn = SAGE(hidden_channels, hidden_channels, drop_rate=drop_rate)
        self.gnn = to_hetero(gnn, metadata, aggr='sum')        

        # Group rating predictor
        self.predictor = Linear(-1, num_node_info['item'])

        # User rating predictor
        self.predictor_user = Linear(-1, num_node_info['item'])
        self.mseloss = nn.MSELoss()


    def forward(self, x, edge_index):        
        emb_x = copy.copy(x)
        if self.training:            
            for node_type in self.emb:                
                emb_x[node_type] = self.emb[node_type](x[node_type]) 
                if node_type == 'group':
                    emb_x[node_type] = emb_x[node_type] * 0
                else:
                    emb_x[node_type] = F.dropout(emb_x[node_type], self.drop_rate)
            
            rep = self.gnn(emb_x, edge_index)
            out = {}
            rep_group = F.dropout(rep['group'], self.drop_rate)
            out['group'] = self.predictor(rep_group)
            rep_user = F.dropout(rep['user'], self.drop_rate)
            out['user'] = self.predictor(rep_user)

            return out
        else:
            for node_type in self.emb:                
                emb_x[node_type] = self.emb[node_type](x[node_type]) 
                if node_type == 'group':
                    emb_x[node_type] = emb_x[node_type] * 0
            rep = self.gnn(emb_x, edge_index)
            out = {}
            out['group'] = self.predictor(rep['group'])

            return out


class RandGR(torch.nn.Module):
    def __init__(self, hidden_channels, metadata, num_node_info, gnn_type, device, drop_rate=0.2):
        super().__init__()
        self.device = device
        self.drop_rate = drop_rate

        # Embedding
        self.emb = {}
        for key in num_node_info:
            self.emb[key] = nn.Embedding(num_node_info[key], hidden_channels).to(device)

        # GNN for inductive learning
        gnn = GAT(hidden_channels, hidden_channels, drop_rate=drop_rate)
        self.gnn = to_hetero(gnn, metadata, aggr='sum')        
        # Group rating predictor
        self.predictor = Linear(-1, num_node_info['item'])

        # GNN for inductive learning
        rand_gnn = GAT(hidden_channels, hidden_channels, drop_rate=drop_rate)
        self.rand_gnn = to_hetero(rand_gnn, metadata, aggr='sum')        
        # Group rating predictor
        self.rand_predictor = Linear(-1, num_node_info['item'])



    def forward(self, x, edge_index):        
        emb_x = copy.copy(x)
        if self.training:            
            for node_type in self.emb:                
                emb_x[node_type] = self.emb[node_type](x[node_type]) 
                if node_type == 'group':
                    emb_x[node_type] = emb_x[node_type] * 0
                else:
                    emb_x[node_type] = F.dropout(emb_x[node_type], self.drop_rate)
            
            rep = self.gnn(emb_x, edge_index)
            out = {}
            rep_group = F.dropout(rep['group'], self.drop_rate)
            out['group'] = self.predictor(rep_group)
            rep_user = F.dropout(rep['user'], self.drop_rate)
            out['user'] = self.predictor(rep_user)

            return out
        else:
            for node_type in self.emb:                
                emb_x[node_type] = self.emb[node_type](x[node_type]) 
                if node_type == 'group':
                    emb_x[node_type] = emb_x[node_type] * 0
            rep = self.gnn(emb_x, edge_index)
            out = {}
            out['group'] = self.predictor(rep['group'])

            return out

