import torch
from torch_geometric.data import Data, HeteroData
from typing import List, Optional, Tuple, Literal, Union

from graphplace.models import Benchmark

def parse_netlist_pb(file_path: str) -> Tuple[List[str], List[torch.Tensor], List[dict]]:
    """
    Parses a netlist.pb.txt file to extract node names and connectivity.
    """
    node_names = []
    node_name_to_idx = {}
    node_metadata = []
    
    with open(file_path, 'r') as f:
        current_node = None
        for line in f:
            line = line.strip()
            if line.startswith('node {'):
                current_node = {}
            elif line.startswith('name:') and current_node is not None:
                name = line.split(':', 1)[1].strip().strip('"')
                if name == "__metadata__":
                    current_node = None
                    continue
                current_node['name'] = name
                node_names.append(name)
                node_name_to_idx[name] = len(node_names) - 1
            elif line.startswith('}'):
                if current_node and 'name' in current_node:
                    node_metadata.append(current_node)
                current_node = None
    
    net_nodes = []
    with open(file_path, 'r') as f:
        current_node_idx = -1
        current_inputs = []
        for line in f:
            line = line.strip()
            if line.startswith('node {'):
                current_node_idx = -1
                current_inputs = []
            elif line.startswith('name:'):
                name = line.split(':', 1)[1].strip().strip('"')
                if name in node_name_to_idx:
                    current_node_idx = node_name_to_idx[name]
            elif line.startswith('input:'):
                input_str = line.split(':', 1)[1].strip().strip('"')
                source_node_name = input_str.split('/')[0]
                if source_node_name in node_name_to_idx:
                    source_idx = node_name_to_idx[source_node_name]
                    current_inputs.append(source_idx)
            elif line.startswith('}'):
                if current_node_idx != -1 and current_inputs:
                    net_members = [current_node_idx] + current_inputs
                    net_nodes.append(torch.tensor(net_members, dtype=torch.long))
                    
    return node_names, net_nodes, node_metadata

def to_hetero_data(
    benchmark: Benchmark,
    net_nodes: Optional[List[torch.Tensor]] = None,
    net_pin_nodes: Optional[List[torch.Tensor]] = None,
    include_positions: bool = True
) -> HeteroData:
    """
    Converts a Benchmark object into a Bipartite HeteroData object.
    
    If net_nodes is provided, it overrides benchmark.net_nodes.
    """
    data = HeteroData()
    
    # 1. Macro Features
    m_size = benchmark.macro_sizes.clone()
    m_size[:, 0] /= (benchmark.canvas_width if benchmark.canvas_width > 0 else 1.0)
    m_size[:, 1] /= (benchmark.canvas_height if benchmark.canvas_height > 0 else 1.0)
    
    m_fixed = benchmark.macro_fixed.float().unsqueeze(-1)
    
    m_soft = torch.zeros(benchmark.num_macros, 1)
    if benchmark.num_soft_macros > 0:
        m_soft[benchmark.num_hard_macros:] = 1.0
        
    m_pos = benchmark.macro_positions.clone()
    m_pos[:, 0] /= (benchmark.canvas_width if benchmark.canvas_width > 0 else 1.0)
    m_pos[:, 1] /= (benchmark.canvas_height if benchmark.canvas_height > 0 else 1.0)
    
    # [x, y, w, h, fixed, soft]
    x_macro = torch.cat([m_pos, m_size, m_fixed, m_soft], dim=-1)
    data['macro'].x = x_macro
    
    # Use provided net_nodes or benchmark ones
    actual_net_nodes = net_nodes if net_nodes is not None else benchmark.net_nodes
    num_nets = len(actual_net_nodes) if actual_net_nodes else benchmark.num_nets
    
    # 2. Net Features
    if actual_net_nodes:
        net_degrees = torch.tensor([len(net) for net in actual_net_nodes], dtype=torch.float32).unsqueeze(-1)
    else:
        net_degrees = torch.zeros((num_nets, 1), dtype=torch.float32)
        
    net_degrees = torch.log1p(net_degrees)
    
    if benchmark.net_weights.shape[0] == num_nets:
        net_weights = benchmark.net_weights.float().unsqueeze(-1)
    else:
        net_weights = torch.ones((num_nets, 1), dtype=torch.float32)
    
    x_net = torch.cat([net_degrees, net_weights], dim=-1)
    data['net'].x = x_net
    
    # 3. Port Features
    if benchmark.port_positions.shape[0] > 0:
        p_pos = benchmark.port_positions.clone()
        p_pos[:, 0] /= (benchmark.canvas_width if benchmark.canvas_width > 0 else 1.0)
        p_pos[:, 1] /= (benchmark.canvas_height if benchmark.canvas_height > 0 else 1.0)
        data['port'].x = p_pos
    else:
        data['port'].x = torch.zeros((0, 2))

    # 4. Edges
    m2n_src, m2n_dst = [], []
    if actual_net_nodes:
        for net_idx, members in enumerate(actual_net_nodes):
            for macro_idx in members:
                macro_int = int(macro_idx)
                if macro_int < benchmark.num_macros:
                    m2n_src.append(macro_int)
                    m2n_dst.append(net_idx)
            
    if m2n_src:
        data['macro', 'to', 'net'].edge_index = torch.tensor([m2n_src, m2n_dst], dtype=torch.long)
        data['net', 'to', 'macro'].edge_index = torch.tensor([m2n_dst, m2n_src], dtype=torch.long)
    
    # Port <-> Net
    actual_net_pin_nodes = net_pin_nodes if net_pin_nodes is not None else benchmark.net_pin_nodes
    p2n_src, p2n_dst = [], []
    if actual_net_pin_nodes:
        for net_idx, ports in enumerate(actual_net_pin_nodes):
            for port_idx in ports:
                p2n_src.append(int(port_idx))
                p2n_dst.append(net_idx)
                
    if p2n_src:
        data['port', 'to', 'net'].edge_index = torch.tensor([p2n_src, p2n_dst], dtype=torch.long)
        data['net', 'to', 'port'].edge_index = torch.tensor([p2n_dst, p2n_src], dtype=torch.long)
        
    return data

def to_pyg_data(
    benchmark: Benchmark = None,
    netlist_file: str = None,
    expansion: Literal['star', 'clique'] = 'star',
    include_positions: bool = True
) -> Union[Data, HeteroData]:
    """Legacy wrapper for backward compatibility."""
    net_nodes = None
    if netlist_file:
        print(f"Parsing netlist from {netlist_file}...")
        _, net_nodes, _ = parse_netlist_pb(netlist_file)
        
    if benchmark:
        if expansion == 'star':
            return to_hetero_data(benchmark, net_nodes=net_nodes)
    
    return Data(x=torch.zeros((1, 1)))
