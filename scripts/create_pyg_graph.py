import argparse
import sys
import torch
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from graphplace.models import Benchmark
from graphplace.graph.pyg_converter import to_pyg_data

def main():
    parser = argparse.ArgumentParser(description="Convert a Benchmark to a PyG Graph.")
    parser.add_argument("--bench", type=str, default="ibm01", help="Benchmark name.")
    parser.add_argument("--expansion", type=str, choices=["star", "clique"], default="star", help="Hypergraph expansion.")
    parser.add_argument("--data-dir", type=str, default="data/processed/public", help="Directory containing .pt benchmarks.")
    parser.add_argument("--out-dir", type=str, default="data/processed/pyg", help="Directory to save PyG graphs.")
    
    args = parser.parse_args()
    
    bench_file = Path(args.data_dir) / f"{args.bench}.pt"
    if not bench_file.exists():
        print(f"Error: Benchmark file {bench_file} not found.")
        sys.exit(1)
        
    print(f"Loading benchmark: {args.bench}")
    benchmark = Benchmark.load(str(bench_file))
    
    # Try to find corresponding netlist.pb.txt
    netlist_path = None
    
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

    netlist_path = get_netlist_path_local(project_root, args.bench)
    
    if netlist_path and netlist_path.exists():
        print(f"Found raw netlist at: {netlist_path}")
    else:
        print(f"Warning: Could not find raw netlist for {args.bench}. Using metadata only.")
        netlist_path = None

    print(f"Converting to PyG graph using {args.expansion} expansion...")
    pyg_data = to_pyg_data(benchmark, netlist_file=str(netlist_path) if netlist_path else None, expansion=args.expansion)
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_file = out_dir / f"{args.bench}_{args.expansion}.pt"
    torch.save(pyg_data, out_file)
    
    print(f"Graph saved to: {out_file}")
    print(f"Summary:")
    print(f"  Nodes: {pyg_data.num_nodes}")
    print(f"  Edges: {pyg_data.num_edges}")
    print(f"  Node features: {pyg_data.x.shape}")

if __name__ == "__main__":
    main()
