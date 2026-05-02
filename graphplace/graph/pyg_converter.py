import torch
from torch_geometric.data import Data
from typing import List, Optional, Tuple, Literal

# Try to import Benchmark from locally available modules
try:
    from graphplace.models import Benchmark
except ImportError:
    try:
        from models import Benchmark
    except ImportError:
        Benchmark = None

def parse_netlist_pb(file_path: str) -> Tuple[List[str], List[torch.Tensor], List[dict]]:
    """
    Parses a netlist.pb.txt file to extract node names and connectivity.
    
    Returns:
        A tuple of (node_names, net_nodes, metadata_list)
    """
    node_names = []
    node_name_to_idx = {}
    node_metadata = []
    
    # First pass: collect all node names and their indices
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
    
    # Second pass: collect inputs (connectivity)
    # We treat each node's inputs as a net where the node is a sink and inputs are sources.
    # Alternatively, we can group by net if we can identify them.
    # In .pb.txt, 'input' points to 'NodeName/PinName'.
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
                    # In this format, each node with inputs defines a "net" 
                    # where the node itself and its sources are connected.
                    net_members = [current_node_idx] + current_inputs
                    net_nodes.append(torch.tensor(net_members, dtype=torch.long))
                    
    return node_names, net_nodes, node_metadata

def to_pyg_data(
    benchmark: 'Benchmark' = None,
    netlist_file: str = None,
    expansion: Literal['star', 'clique'] = 'star',
    include_positions: bool = True
) -> Data:
    """
    Converts a Benchmark object or a netlist file into a PyTorch Geometric Data object.
    
    Args:
        benchmark: The Benchmark object containing netlist data.
        netlist_file: Path to a netlist.pb.txt file (used if benchmark is None or has no nets).
        expansion: The hypergraph expansion method ('star' or 'clique').
        include_positions: Whether to include macro positions in node features.
        
    Returns:
        A torch_geometric.data.Data object.
    """
    if benchmark is None and netlist_file is None:
        raise ValueError("Either benchmark or netlist_file must be provided.")
    
    # If benchmark has no nets, try loading from netlist_file
    if (benchmark is None or len(benchmark.net_nodes) == 0) and netlist_file:
        print(f"Parsing netlist from {netlist_file}...")
        _, net_nodes, _ = parse_netlist_pb(netlist_file)
    else:
        net_nodes = benchmark.net_nodes
        
    if benchmark is None:
        # Create a basic graph if no benchmark metadata is provided
        num_macros = len(set([idx.item() for net in net_nodes for idx in net]))
        x = torch.zeros((num_macros, 4)) # Dummy features
        num_nets = len(net_nodes)
        net_weights = torch.ones(num_nets)
    else:
        num_macros = benchmark.num_macros
        num_nets = len(net_nodes)
        net_weights = benchmark.net_weights if len(benchmark.net_weights) == num_nets else torch.ones(num_nets)
        
        # Prepare Node Features (x)
        # Features: [width, height, is_fixed, is_soft]
        macro_sizes = benchmark.macro_sizes / (benchmark.canvas_width if benchmark.canvas_width > 0 else 1.0)
        is_fixed = benchmark.macro_fixed.float().unsqueeze(-1)
        
        if hasattr(benchmark, 'num_hard_macros'):
            is_soft = torch.zeros(num_macros, 1)
            is_soft[benchmark.num_hard_macros:] = 1.0
        else:
            is_soft = torch.zeros(num_macros, 1)
        
        features = [macro_sizes, is_fixed, is_soft]
        
        if include_positions:
            pos_norm = benchmark.macro_positions.clone()
            pos_norm[:, 0] /= (benchmark.canvas_width if benchmark.canvas_width > 0 else 1.0)
            pos_norm[:, 1] /= (benchmark.canvas_height if benchmark.canvas_height > 0 else 1.0)
            features.insert(0, pos_norm)
            
        x_macros = torch.cat(features, dim=-1)
        x = x_macros

    # prepare Edges (edge_index)
    if expansion == 'star':
        num_nodes = x.size(0) + num_nets
        x_nets = torch.zeros((num_nets, x.size(1)))
        x = torch.cat([x, x_nets], dim=0)
        
        sources = []
        targets = []
        edge_weights = []
        
        for net_idx, members in enumerate(net_nodes):
            weight = net_weights[net_idx]
            net_node_idx = num_macros + net_idx
            
            for macro_idx in members:
                # Add bi-directional edges
                sources.append(int(macro_idx))
                targets.append(net_node_idx)
                edge_weights.append(float(weight))
                
                sources.append(net_node_idx)
                targets.append(int(macro_idx))
                edge_weights.append(float(weight))
                
        edge_index = torch.tensor([sources, targets], dtype=torch.long)
        edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(-1)
        
    elif expansion == 'clique':
        sources = []
        targets = []
        edge_weights = []
        
        for net_idx, members in enumerate(net_nodes):
            weight = net_weights[net_idx]
            nodes = members.tolist()
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    sources.append(nodes[i])
                    targets.append(nodes[j])
                    edge_weights.append(float(weight))
                    sources.append(nodes[j])
                    targets.append(nodes[i])
                    edge_weights.append(float(weight))
                    
        edge_index = torch.tensor([sources, targets], dtype=torch.long)
        edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(-1)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
