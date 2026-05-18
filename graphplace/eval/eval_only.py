import os
import sys
import torch
import numpy as np
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from graphplace.rl.placement_env import PlacementEnv
from graphplace.gnn_placer import PlaceGNN
from graphplace.graph.pyg_converter import to_hetero_data, parse_netlist_pb
from graphplace.models import Benchmark

def run_eval(bench_name, model_path, device_str="cuda"):
    print(f"\n--- EVALUATING MODEL: {model_path} ---")
    
    # 0. Device setup
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Setup Env and Model
    env = PlacementEnv(benchmark_name=bench_name)
    model = PlaceGNN()
    
    # 2. Check for warm-start
    warm_start_path = Path("output") / bench_name / f"{bench_name}_legalized.pt"
    warm_start_pos = torch.load(warm_start_path) if warm_start_path.exists() else None
    
    # 3. Load Graph Data
    bench_pt = f"data/processed/public/{bench_name}.pt"
    benchmark_data = Benchmark.load(bench_pt)
    netlist_path = f"externals/MacroPlacement/Testcases/ICCAD04/{bench_name}/netlist.pb.txt"
    _, net_nodes, _ = parse_netlist_pb(netlist_path)
    base_graph_data = to_hetero_data(benchmark_data, net_nodes=net_nodes).to(device)

    # 4. Load the specific version
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded: {model_path}")
    else:
        print(f"Error: {model_path} does not exist!")
        return

    model.to(device)
    model.eval()

    # 5. Run 1 sequence
    obs = env.reset(options={'warm_start_pos': warm_start_pos} if warm_start_pos is not None else None)
    initial_score = env.last_score
    print(f"Start Score: {initial_score:.4f}")
    
    with torch.no_grad():
        for i in range(10):  # Reduced to 10 steps for faster CPU evaluation
            step_graph_data = base_graph_data.clone()
            new_pos = torch.tensor(obs[:, :2], device=device, dtype=torch.float32)
            step_graph_data['macro'].x = torch.cat([new_pos, step_graph_data['macro'].x[:, 2:]], dim=1)
            
            # KNN for proximity edges
            k = min(5, new_pos.size(0) - 1)
            try:
                from torch_cluster import knn_graph
                edge_index = knn_graph(new_pos, k, loop=False)
                step_graph_data['macro', 'near', 'macro'].edge_index = edge_index
            except Exception:
                dist = torch.cdist(new_pos, new_pos)
                dist.fill_diagonal_(float('inf'))
                topk = dist.topk(k, largest=False)
                indices = topk.indices
                src = torch.arange(new_pos.size(0), device=device).unsqueeze(1).expand(-1, k).reshape(-1)
                dst = indices.reshape(-1)
                step_graph_data['macro', 'near', 'macro'].edge_index = torch.stack([src, dst], dim=0)

            _, mu = model(step_graph_data)
            action = mu.cpu().numpy()
            obs, reward, done, info = env.step(action, fast=True) 
            print(f"Step {i+1}/10 | Proxy: {info['proxy_score']:.4f}")
            if done: break
            
    # Final Precise Score
    print("\nCalculating Final Precise Score (including congestion and density)...")
    final_score = env._get_score(env.current_pos, fast=False)
    improvement = initial_score - final_score
    print(f"Start Score: {initial_score:.4f}")
    print(f"End Score:   {final_score:.4f}")
    print(f"Net Gain:    {improvement:.4f}")

    # 6. Save the final placement
    output_dir = Path("output") / bench_name
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"{bench_name}_gnn_final.pt"
    torch.save(env.current_pos.cpu(), save_path)
    print(f"Final placement saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--bench", type=str, default="ibm01")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    run_eval(args.bench, args.model, args.device)
