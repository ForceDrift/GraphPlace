import argparse
import sys
import torch
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from graphplace.utils.vis.vis import plot_graph
from types import SimpleNamespace

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
    
    data = torch.load(str(bench_file), weights_only=False)
    benchmark = SimpleNamespace(**data)
    
    # Try to find corresponding netlist.pb.txt to get connectivity
    
    # Re-using the logic from create_pyg_graph
    def get_netlist_path_local(project_root: Path, name: str) -> Path:
        externals_root = project_root / "externals" / "MacroPlacement"
        if name.startswith("ibm"):
            # strip _dreamplace if present to find the original netlist
            real_name = name.split('_')[0]
            return externals_root / "Testcases" / "ICCAD04" / real_name / "netlist.pb.txt"
        
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
            import re
            all_node_names = []
            all_net_nodes = []
            
            with open(netlist_path, 'r', encoding='utf-8') as f:
                in_node = False
                current_name = None
                inputs = []
                for line in f:
                    line = line.strip()
                    if line == 'node {':
                        in_node = True
                        current_name = None
                        inputs = []
                    elif line == '}':
                        if in_node and current_name and current_name != '__metadata__':
                            idx = len(all_node_names)
                            all_node_names.append(current_name)
                            if inputs:
                                all_net_nodes.append([idx] + inputs) # We will resolve input names later
                        in_node = False
                    elif in_node:
                        m = re.match(r'name:\s*"(.*)"', line)
                        if m: current_name = m.group(1)
                        m = re.match(r'input:\s*"(.*)"', line)
                        if m: inputs.append(m.group(1))
            
            # Resolve input names to indices
            name_to_idx_global = {name: i for i, name in enumerate(all_node_names)}
            resolved_nets = []
            for net in all_net_nodes:
                resolved = [net[0]] # The sink
                for src in net[1:]:
                    src_node = src.split('/')[0]
                    if src_node in name_to_idx_global:
                        resolved.append(name_to_idx_global[src_node])
                resolved_nets.append(resolved)
            all_net_nodes = resolved_nets
            
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
