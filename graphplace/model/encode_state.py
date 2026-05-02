import torch
from torch_geometric.data import Data
from typing import Optional, Literal

try:
    from graphplace.models import Benchmark
    from graphplace.graph.pyg_converter import to_pyg_data
except ImportError:
    pass

class StateEncoder:
    """
    Encodes the current placement state (graph topology + geometric + wirelength features)
    into a PyTorch Geometric Data object for GNN or RL models.
    """
    def __init__(self, benchmark: 'Benchmark', expansion: Literal['star', 'clique'] = 'star'):
        """
        Initializes the state encoder for a given benchmark.
        
        Args:
            benchmark: The Benchmark object containing the netlist and constraints.
            expansion: Hypergraph expansion method to use.
        """
        self.benchmark = benchmark
        self.expansion = expansion
        
        # Pre-compute static graph features (e.g., node degrees)
        self.num_macros = benchmark.num_macros
        self.num_nets = len(benchmark.net_nodes) if hasattr(benchmark, 'net_nodes') and benchmark.net_nodes else benchmark.num_nets
        
        self.macro_degrees = torch.zeros(self.num_macros)
        if hasattr(benchmark, 'net_nodes') and benchmark.net_nodes:
            for net in benchmark.net_nodes:
                for src in net:
                    if src < self.num_macros:
                        self.macro_degrees[src] += 1
                        
        # Normalize degrees
        max_deg = self.macro_degrees.max()
        if max_deg > 0:
            self.macro_degrees_norm = self.macro_degrees / max_deg
        else:
            self.macro_degrees_norm = self.macro_degrees

    def encode(self, current_positions: Optional[torch.Tensor] = None) -> Data:
        """
        Encodes the current placement graph and shapes into a PyG Data object.
        
        Args:
            current_positions: Optional [num_macros, 2] tensor of current coordinates.
                               If None, uses benchmark.macro_positions.
                               
        Returns:
            A torch_geometric.data.Data object encoding the state.
            Features (x) include:
                - Normalized position (x, y)
                - Normalized dimensions (w, h)
                - Fixed macro flag
                - Soft macro flag
                - Normalized node degree
                - Normalized Bounding Box spans (dx, dy) for net-nodes (if star expansion)
        """
        # Use temporary positions if provided
        orig_positions = None
        if current_positions is not None:
            orig_positions = self.benchmark.macro_positions
            self.benchmark.macro_positions = current_positions
            
        # Base conversion: yields [pos_x, pos_y, w, h, is_fixed, is_soft]
        try:
            data = to_pyg_data(self.benchmark, expansion=self.expansion, include_positions=True)
        finally:
            if orig_positions is not None:
                self.benchmark.macro_positions = orig_positions

        num_nodes = data.x.size(0)

        degree_feature = torch.zeros(num_nodes, 1)
        degree_feature[:self.num_macros, 0] = self.macro_degrees_norm
        
        bbox_feature = torch.zeros(num_nodes, 2)
        
        positions = current_positions if current_positions is not None else self.benchmark.macro_positions
        canvas_w = self.benchmark.canvas_width if self.benchmark.canvas_width > 0 else 1.0
        canvas_h = self.benchmark.canvas_height if self.benchmark.canvas_height > 0 else 1.0

        if self.expansion == 'star' and hasattr(self.benchmark, 'net_nodes') and self.benchmark.net_nodes:
            for net_idx, net in enumerate(self.benchmark.net_nodes):
                if len(net) > 0:
                    net_pos = positions[net]
                    min_pos = net_pos.min(dim=0)[0]
                    max_pos = net_pos.max(dim=0)[0]
                    span = max_pos - min_pos
                    
                    span[0] /= canvas_w
                    span[1] /= canvas_h
                    
                    net_node_idx = self.num_macros + net_idx
                    if net_node_idx < num_nodes:
                        bbox_feature[net_node_idx] = span
                        
                        center = (min_pos + max_pos) / 2
                        data.x[net_node_idx, 0] = center[0] / canvas_w
                        data.x[net_node_idx, 1] = center[1] / canvas_h

        data.x = torch.cat([data.x, degree_feature, bbox_feature], dim=-1)
        
        return data

def encode_state(benchmark: 'Benchmark', current_positions: Optional[torch.Tensor] = None, expansion: Literal['star', 'clique'] = 'star') -> Data:
    """
    Convenience function to encode a placement state into a PyG graph tensor.
    """
    encoder = StateEncoder(benchmark, expansion=expansion)
    return encoder.encode(current_positions)
