import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATv2Conv, global_mean_pool, GlobalAttention
from typing import Dict, Tuple

class PlaceGNN(nn.Module):
    """
    Heterogeneous GNN for Macro Placement.
    Uses Graph Attention (GAT) to propagate connectivity constraints.
    """
    def __init__(
        self,
        macro_in_channels: int = 6,  # [x, y, w, h, fixed, soft]
        net_in_channels: int = 2,    # [degree, weight]
        port_in_channels: int = 2,   # [x, y]
        hidden_channels: int = 64,
        num_layers: int = 3,
        grid_size: int = 32          # 32x32 grid for global logits
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.grid_size = grid_size
        
        # 1. Input Projections
        self.macro_lin = nn.Linear(macro_in_channels, hidden_channels)
        self.net_lin = nn.Linear(net_in_channels, hidden_channels)
        self.port_lin = nn.Linear(port_in_channels, hidden_channels)
        
        # 2. Hetero Convolution Layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('macro', 'to', 'net'): GATv2Conv(hidden_channels, hidden_channels // 4, heads=4, add_self_loops=False),
                ('net', 'to', 'macro'): GATv2Conv(hidden_channels, hidden_channels // 4, heads=4, add_self_loops=False),
                ('port', 'to', 'net'): GATv2Conv(hidden_channels, hidden_channels // 4, heads=4, add_self_loops=False),
                ('net', 'to', 'port'): GATv2Conv(hidden_channels, hidden_channels // 4, heads=4, add_self_loops=False),
            }, aggr='sum')
            self.convs.append(conv)
            
        # 3. Global Context Substream
        self.global_att = GlobalAttention(nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        ))
        
        # 4. Output Heads
        # Global Grid Logits (32x32 = 1024)
        self.grid_head = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, grid_size * grid_size)
        )
        
        # Local Offset Delta [dx, dy]
        self.offset_head = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 2),
            nn.Tanh() # Clamp to [-1, 1] relative units
        )

    def forward(self, data):
        # x_dict contains node features by type
        x_dict = {
            'macro': self.macro_lin(data['macro'].x),
            'net': self.net_lin(data['net'].x),
            'port': self.port_lin(data['port'].x) if data['port'].num_nodes > 0 else torch.zeros((0, self.hidden_channels))
        }
        
        # Apply Convolutions
        for conv in self.convs:
            x_dict = conv(x_dict, data.edge_index_dict)
            x_dict = {key: F.elu(x) for key, x in x_dict.items()}
            
        # Compute Global Context from macros
        g_context = self.global_att(x_dict['macro']) # [1, hidden]
        g_context_expanded = g_context.expand(x_dict['macro'].size(0), -1) # [N_macros, hidden]
        
        # Concatenate macro features with global context
        out_features = torch.cat([x_dict['macro'], g_context_expanded], dim=-1)
        
        # Grid Logits
        grid_logits = self.grid_head(out_features)
        
        # Offsets
        offsets = self.offset_head(out_features)
        
        return grid_logits, offsets
