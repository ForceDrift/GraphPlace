import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import os

class BipartiteGNNLayer(nn.Module):
    """
    A simple bipartite message passing layer for macro-net graphs.
    """
    def __init__(self, macro_dim, net_dim, hidden_dim):
        super().__init__()
        # Macro to Net
        self.m2n = nn.Linear(macro_dim, hidden_dim)
        # Net to Macro
        self.n2m = nn.Linear(net_dim + hidden_dim, hidden_dim)
        # Final updates
        self.macro_update = nn.Linear(macro_dim + hidden_dim, hidden_dim)
        self.net_update = nn.Linear(net_dim + hidden_dim, hidden_dim)

    def forward(self, macro_feats, net_feats, adj_m2n, adj_n2m):
        # 1. Macros -> Nets
        # adj_m2n is [num_nets, num_macros]
        macro_msgs = self.m2n(macro_feats)
        net_agg = torch.matmul(adj_m2n, macro_msgs) # Aggregated macro info per net
        
        # 2. Update Net features
        net_combined = torch.cat([net_feats, net_agg], dim=-1)
        new_net_feats = F.relu(self.net_update(net_combined))
        
        # 3. Nets -> Macros
        net_msgs = self.n2m(torch.cat([net_feats, net_agg], dim=-1)) # Use updated info
        macro_agg = torch.matmul(adj_n2m, net_msgs) # Aggregated net info per macro
        
        # 4. Update Macro features
        macro_combined = torch.cat([macro_feats, macro_agg], dim=-1)
        new_macro_feats = F.relu(self.macro_update(macro_combined))
        
        return new_macro_feats, new_net_feats

class MacroPlacementGNN(nn.Module):
    def __init__(self, num_layers=3, hidden_dim=64):
        super().__init__()
        # Initial encoders
        self.macro_encoder = nn.Linear(5, hidden_dim) # [w, h, a, x, y]
        self.net_encoder = nn.Linear(3, hidden_dim)   # [num_pins, bb_w, bb_h]
        
        self.layers = nn.ModuleList([
            BipartiteGNNLayer(hidden_dim, hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Policy head: Predict local movement logits (e.g., 7x7 grid around current pos)
        self.grid_size = 7
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.grid_size * self.grid_size) 
        )
        
        # Value head (Critic): Predict state value
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, macro_feats, net_feats, adj_m2n, adj_n2m):
        h_m = F.relu(self.macro_encoder(macro_feats))
        h_n = F.relu(self.net_encoder(net_feats))
        
        for layer in self.layers:
            h_m, h_n = layer(h_m, h_n, adj_m2n, adj_n2m)
            
        # Predict move logits for each macro [num_macros, grid_size * grid_size]
        logits = self.policy_head(h_m)
        
        # Predict state value (scalar)
        value = self.value_head(h_m.mean(dim=0, keepdim=True))
        
        return logits, value

def build_bipartite_adj(num_macros, net_nodes):
    """
    Build adjacency matrices for the bipartite graph.
    """
    num_nets = len(net_nodes)
    adj_m2n = torch.zeros((num_nets, num_macros))
    
    for net_idx, pins in enumerate(net_nodes):
        for macro_idx in pins:
            adj_m2n[net_idx, macro_idx] = 1.0
            
    # Normalize by degree to keep values stable
    m_degrees = adj_m2n.sum(dim=0, keepdim=True).clamp(min=1.0) # [1, num_macros]
    n_degrees = adj_m2n.sum(dim=1, keepdim=True).clamp(min=1.0) # [num_nets, 1]
    
    adj_m2n_norm = adj_m2n / n_degrees # [num_nets, num_macros] / [num_nets, 1] -> broadasts
    adj_n2m_norm = adj_m2n.t() / m_degrees.t() # [num_macros, num_nets] / [num_macros, 1] -> broadasts
    
    return adj_m2n_norm, adj_n2m_norm

def prepare_features(data, current_pos=None):
    """
    Prepare initial features for macros and nets.
    """
    num_macros = data['num_macros']
    sizes = data['macro_sizes']
    if current_pos is None:
        current_pos = data['macro_positions']
        
    # Macro features: [w, h, area, x_norm, y_norm]
    cw, ch = data['canvas_width'], data['canvas_height']
    macro_feats = torch.zeros((num_macros, 5))
    macro_feats[:, 0] = sizes[:, 0] / cw
    macro_feats[:, 1] = sizes[:, 1] / ch
    macro_feats[:, 2] = (sizes[:, 0] * sizes[:, 1]) / (cw * ch)
    macro_feats[:, 3] = current_pos[:, 0] / cw
    macro_feats[:, 4] = current_pos[:, 1] / ch
    
    # Net features: [num_pins, bb_w, bb_h]
    net_nodes = data['net_nodes']
    num_nets = len(net_nodes)
    net_feats = torch.zeros((num_nets, 3))
    
    for i, pins in enumerate(net_nodes):
        pin_pos = current_pos[pins]
        min_p = pin_pos.min(dim=0)[0]
        max_p = pin_pos.max(dim=0)[0]
        bb = max_p - min_p
        net_feats[i, 0] = len(pins) / 100.0 # Normalized
        net_feats[i, 1] = bb[0] / cw
        net_feats[i, 2] = bb[1] / ch
        
    return macro_feats, net_feats

if __name__ == "__main__":
    # Test on ibm01
    pt_path = "data/processed/public/ibm01.pt"
    if os.path.exists(pt_path):
        data = torch.load(pt_path, weights_only=False)
        m_feats, n_feats = prepare_features(data)
        adj_m2n, adj_n2m = build_bipartite_adj(data['num_macros'], data['net_nodes'])
        
        model = MacroPlacementGNN()
        moves, value = model(m_feats, n_feats, adj_m2n, adj_n2m)
        
        print(f"GNN Output Shapes:")
        print(f"  Moves: {moves.shape}")
        print(f"  Value: {value.item():.4f}")
        print(f"Sample move for Macro 0: {moves[0].detach().numpy()}")
