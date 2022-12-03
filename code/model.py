import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import MessagePassing, GraphNorm
from torch_scatter import scatter

from torch_scatter import scatter_add
from torch_geometric.utils import softmax
import math

class Conv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim=6, aggr='add'): 
        super().__init__(aggr=aggr)  
        self.aggr = aggr
        self.lin_neg = nn.Linear(in_channels+edge_dim, out_channels) 
        self.lin_root = nn.Linear(in_channels, out_channels)
        
    def forward(self, x, edge_index, edge_attr):
          
        x_adj = torch.cat([x[edge_index[1]],edge_attr], dim=1) 
        #print('x_adj shape:',x_adj.shape)
        x_adj = F.tanh(self.lin_neg(x_adj))
        
        neg_sum = scatter(x_adj, edge_index[0], dim=0, reduce=self.aggr) 
        
        x_out = F.tanh(self.lin_root(x)) + neg_sum
        #x_out = self.bn1(x_out)
        return x_out
    
"""
############### GlobalAttentaion###############
"""
def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()
            
    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)

class GlobalAttention(torch.nn.Module):
    def __init__(self, gate_nn, nn=None):
        super().__init__()
        self.gate_nn = gate_nn
        self.nn = nn
        self.reset_parameters()
        
    def reset_parameters(self):
        reset(self.gate_nn)
        reset(self.nn)
    
    def forward(self, x, batch, size=None):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        size = batch[-1].item() + 1 if size is None else size
        
        gate = self.gate_nn(x).view(-1, 1)
        x = self.nn(x) if self.nn is not None else x
        assert gate.dim() == x.dim() and gate.size(0) == x.size(0)
        
        gate = softmax(gate, batch, num_nodes=size)
        out = scatter_add(gate * x, batch, dim=0, dim_size=size)
        
        return out, gate
    
    def __repr__(self) -> str:
        return(f'{self.__class__.__name__}(gate_nn={self.gate_nn}, '
               f'nn={self.nn})')

class CCPGraph(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv(39, 64)  
        self.gn1 = GraphNorm(64)
        self.conv2 = Conv(64, 16)
        self.gn2 = GraphNorm(16)
        
        #pool
        gate_nn = nn.Sequential(nn.Linear(16,64),
                                nn.ReLU(),
                                nn.Linear(64,32),
                                nn.ReLU(),
                                nn.Linear(32, 1))
        
        self.readout = GlobalAttention(gate_nn)
        self.lin1 = nn.Linear(16, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.lin2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dp2 = nn.Dropout(p=0.4)
        self.lin3 = nn.Linear(512, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dp3 = nn.Dropout(p=0.25)
        self.lin = nn.Linear(128,1)
        
        
    def forward(self, data):
        x = self.conv1(data.x, data.edge_index, data.edge_attr)
        x = self.conv2(x, data.edge_index, data.edge_attr)
        
        embedding, att = self.readout(x, data.batch)
        #print('embedding shape',embedding.shape)
        
        out = self.dp1(self.bn1(F.relu(self.lin1(embedding)))) 
        out = self.dp2(self.bn2(F.relu(self.lin2(out)))) 
        out = self.dp3(self.bn3(F.relu(self.lin3(out))))
        out = self.lin(out)
        
        return out.view(-1), att