import torch
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from graphplace.models import Benchmark
from graphplace.graph.pyg_converter import to_pyg_data

def test_pyg_converter():
    print("Running verification for PyG converter...")
    
    # 1. Create a dummy benchmark
    num_macros = 3
    num_nets = 1
    
    # Simple net connecting all 3 macros
    net_nodes = [torch.tensor([0, 1, 2])]
    net_weights = torch.tensor([1.0])
    
    benchmark = Benchmark(
        name="dummy",
        canvas_width=100.0,
        canvas_height=100.0,
        num_macros=num_macros,
        macro_positions=torch.tensor([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]]),
        macro_sizes=torch.tensor([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]]),
        macro_fixed=torch.tensor([False, False, True]),
        macro_names=["m1", "m2", "m3"],
        num_nets=num_nets,
        net_nodes=net_nodes,
        net_weights=net_weights,
        grid_rows=10,
        grid_cols=10,
        num_hard_macros=3,
        num_soft_macros=0
    )
    
    # 2. Test Star Expansion
    print("Testing Star Expansion...")
    data_star = to_pyg_data(benchmark, expansion='star')
    
    expected_nodes = num_macros + num_nets # 3 macros + 1 net = 4 nodes
    assert data_star.num_nodes == expected_nodes, f"Expected {expected_nodes} nodes, got {data_star.num_nodes}"
    
    # Each macro connects to the net node (2 edges per connection for bi-directional)
    # 3 macros * 2 edges = 6 edges
    assert data_star.num_edges == 6, f"Expected 6 edges, got {data_star.num_edges}"
    
    # 3. Test Clique Expansion
    print("Testing Clique Expansion...")
    data_clique = to_pyg_data(benchmark, expansion='clique')
    
    assert data_clique.num_nodes == num_macros # 3 nodes
    # A clique of 3 nodes has 3 connections, bi-directional = 6 edges
    assert data_clique.num_edges == 6, f"Expected 6 edges, got {data_clique.num_edges}"
    
    print("Verification successful!")

if __name__ == "__main__":
    test_pyg_converter()
