import argparse
import sys
import torch
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from graphplace.models import Benchmark
from graphplace.graph.pyg_converter import parse_netlist_pb
from graphplace.utils.vis import plot_graph

def main():
    parser = argparse.ArgumentParser(description="Visualize a Benchmark graph (macros + wires).")
    parser.add_argument("--bench", type=str, default="ibm01", help="Benchmark name.")
    parser.add_argument("--data-dir", type=str, default="data/processed/public", help="Directory containing .pt benchmarks.")
    parser.add_argument("--out", type=str, default="graph_vis_ibm01.png", help="Output image path.")
    parser.add_argument("--max-edges", type=int, default=20000, help="Max edges to draw.")
    
    args = parser.parse_args()
    
    bench_file = Path(args.data_dir) / f"{args.bench}.pt"
    if not bench_file.exists():
        print(f"Error: Benchmark file {bench_file} not found.")
        sys.exit(1)
        
    print(f"Loading benchmark: {args.bench}")
    benchmark = Benchmark.load(str(bench_file))
    
    # Try to find corresponding netlist.pb.txt to get connectivity
    
    # Re-using the logic from create_pyg_graph
    def get_netlist_path_local(project_root: Path, name: str) -> Path:
        externals_root = project_root / "externals" / "MacroPlacement"
        if name.startswith("ibm"):
            return externals_root / "Testcases" / "ICCAD04" / name / "netlist.pb.txt"
        
        block = None
        if "ariane133" in name: block = "ariane133"
        elif "ariane136" in name: block = "ariane136"
        elif "mempool_tile" in name: block = "mempool_tile"
        elif "nvdla" in name: block = "nvdla"
        elif "bp_quad" in name: block = "bp_quad"
        if block is None: return None
        tech = "ASAP7" if "asap7" in name else ("NanGate45" if "ng45" in name else None)
        if tech is None: return None
        return (externals_root / "Flows" / tech / block / "netlist" / "output_CT_Grouping" / "netlist.pb.txt")

    # If the benchmark already contains net_nodes, we don't need to parse the netlist again
    if hasattr(benchmark, 'net_nodes') and benchmark.net_nodes and len(benchmark.net_nodes) > 0:
        print(f"Using {len(benchmark.net_nodes)} nets already present in the benchmark file.")
    else:
        netlist_path = get_netlist_path_local(project_root, args.bench)
        
        if netlist_path and netlist_path.exists():
            print(f"Parsing connectivity from {netlist_path}...")
            all_node_names, all_net_nodes, _ = parse_netlist_pb(str(netlist_path))
            
            # Mapping from netlist node names to benchmark indices
            name_to_idx = {name: i for i, name in enumerate(benchmark.macro_names)}
            
            filtered_net_nodes = []
            for net in all_net_nodes:
                # For each net in the pb.txt, find which macros are in it
                macro_indices = []
                for node_idx in net:
                    node_name = all_node_names[node_idx]
                    if node_name in name_to_idx:
                        macro_indices.append(name_to_idx[node_name])
                
                if len(macro_indices) >= 2:
                    filtered_net_nodes.append(torch.tensor(macro_indices, dtype=torch.long))
            
            benchmark.net_nodes = filtered_net_nodes
            print(f"Filtered to {len(filtered_net_nodes)} nets connecting macros.")
        else:
            print(f"Warning: No netlist found for {args.bench}. Only macros will be plotted.")
            benchmark.net_nodes = []

    print(f"Generating visualization...")
    plot_graph(benchmark, output_path=args.out, max_edges=args.max_edges)

if __name__ == "__main__":
    main()
