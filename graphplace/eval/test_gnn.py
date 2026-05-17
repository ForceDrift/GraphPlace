import os
import sys
from pathlib import Path
import torch
import argparse
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from graphplace.rl.placement_env import PlacementEnv
from graphplace.gnn_placer import PlaceGNN
from graphplace.graph.pyg_converter import to_hetero_data, parse_netlist_pb
from graphplace.models import Benchmark

def test_gnn():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", type=str, default="ibm01")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Environment & Model
    env = PlacementEnv(benchmark_name=args.bench)
    model = PlaceGNN().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    # 2. Load Graph Data
    bench_pt = f"data/processed/public/{args.bench}.pt"
    benchmark = Benchmark.load(bench_pt)
    netlist_path = f"externals/MacroPlacement/Testcases/ICCAD04/{args.bench}/netlist.pb.txt"
    _, net_nodes, _ = parse_netlist_pb(netlist_path)
    graph_data = to_hetero_data(benchmark, net_nodes=net_nodes).to(device)

    # 3. Inference (Greedy Selection - take mu instead of sample)
    print(f"Running inference on {args.bench}...")
    
    warm_start_path = Path("output") / args.bench / f"{args.bench}_legalized.pt"
    options = None
    if warm_start_path.exists():
        options = {'warm_start_pos': torch.load(warm_start_path)}
        
    obs = env.reset(options=options)
    
    # Update graph features with environment state
    new_pos = torch.tensor(obs[:, :2], device=device, dtype=torch.float32)
    graph_data['macro'].x = torch.cat([new_pos, graph_data['macro'].x[:, 2:]], dim=1)
    
    # Construct proximity edges (k=5)
    with torch.no_grad():
        dist = torch.cdist(new_pos, new_pos)
        dist.fill_diagonal_(float('inf'))
        k = min(5, new_pos.size(0) - 1)
        topk = dist.topk(k, largest=False)
        indices = topk.indices
        src = torch.arange(new_pos.size(0), device=device).unsqueeze(1).expand(-1, k).reshape(-1)
        dst = indices.reshape(-1)
        graph_data['macro', 'near', 'macro'].edge_index = torch.stack([src, dst], dim=0)

    with torch.no_grad():
        _, mu = model(graph_data)
    
    # 4. Step Environment & Legalize
    # We take the nudge predicted by the model
    action_np = mu.cpu().numpy()
    next_obs, reward, done, info = env.step(action_np)

    print(f"Inference complete.")
    print(f"Legalized Proxy Score: {info.get('proxy_score', 0):.4f}")

    # 5. Save placement as .pt for competition evaluator
    # Use env.current_pos which holds un-normalized absolute coordinates!
    pos_tensor = env.current_pos.clone().cpu()
    output_path = args.output or f"output/{args.bench}_gnn_result.pt"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(pos_tensor, output_path)
    print(f"Placement saved to {output_path}")

if __name__ == "__main__":
    test_gnn()
