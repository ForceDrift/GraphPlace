import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import argparse

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from graphplace.rl.placement_env import PlacementEnv
from graphplace.gnn_placer import PlaceGNN
from graphplace.graph.pyg_converter import to_hetero_data
from graphplace.models import Benchmark
from graphplace.graph.pyg_converter import parse_netlist_pb

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", nargs='+', default=["ibm01", "ibm02", "ibm03", "ibm04"], help="List of benchmarks to train on")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--steps_per_epoch", type=int, default=50)
    parser.add_argument("--device", type=str, default=None, help="Device to use (cpu, cuda). Auto-detects if not set.")
    args = parser.parse_args()

    # Device setup
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Shared GNN model
    model = PlaceGNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Initialize environments and graphs for all benchmarks
    envs = {}
    graphs = {}
    best_proxies = {}
    
    print(f"Loading environments and graphs for: {args.benchmarks}")
    for b in args.benchmarks:
        print(f"  Loading {b}...")
        envs[b] = PlacementEnv(benchmark_name=b)
        best_proxies[b] = float('inf')
        
        bench_pt = f"data/processed/public/{b}.pt"
        benchmark = Benchmark.load(bench_pt)
        netlist_path = f"externals/MacroPlacement/Testcases/ICCAD04/{b}/netlist.pb.txt"
        _, net_nodes, _ = parse_netlist_pb(netlist_path)
        graphs[b] = to_hetero_data(benchmark, net_nodes=net_nodes).to(device)

    # Training Loop
    for epoch in range(args.epochs):
        # Round-robin: Pick the next benchmark
        bench_name = args.benchmarks[epoch % len(args.benchmarks)]
        env = envs[bench_name]
        base_graph_data = graphs[bench_name]
        
        obs = env.reset()
        print(f"Epoch {epoch} | Bench: {bench_name} | Starting Proxy: {env.last_score:.4f}")
        epoch_reward = 0
        
        log_probs = []
        rewards = []
        
        for step in range(args.steps_per_epoch):
            step_graph_data = base_graph_data.clone()

            new_pos = torch.tensor(obs[:, :2], device=device, dtype=torch.float32)
            step_graph_data['macro'].x = torch.cat([new_pos, step_graph_data['macro'].x[:, 2:]], dim=1)
            
            # Construct proximity edges (k=5)
            with torch.no_grad():
                k = min(5, new_pos.size(0) - 1)
                try:
                    from torch_cluster import knn_graph
                    # knn_graph computes exact KNN incredibly fast without the O(N^2) memory footprint!
                    edge_index = knn_graph(new_pos, k, loop=False)
                    step_graph_data['macro', 'near', 'macro'].edge_index = edge_index
                except Exception as e:
                    print(f"Warning: Failed to import torch_cluster ({e}). Falling back to slow cdist.")
                    dist = torch.cdist(new_pos, new_pos)
                    dist.fill_diagonal_(float('inf'))
                    topk = dist.topk(k, largest=False)
                    indices = topk.indices
                    src = torch.arange(new_pos.size(0), device=device).unsqueeze(1).expand(-1, k).reshape(-1)
                    dst = indices.reshape(-1)
                    step_graph_data['macro', 'near', 'macro'].edge_index = torch.stack([src, dst], dim=0)

            # Forward: Get Mean displacement from GNN
            _, mu = model(step_graph_data)
            
            # Policy: Normal distribution for exploration
            std = torch.ones_like(mu) * 0.05
            dist = Normal(mu, std)
            
            # Sample action
            action_tensor = dist.sample()
            log_prob = dist.log_prob(action_tensor).sum(dim=-1) # [num_macros]
            
            # Environment step
            action_np = action_tensor.cpu().detach().numpy()
            next_obs, reward, done, info = env.step(action_np)
            
            log_probs.append(log_prob)
            rewards.append(torch.tensor(reward, device=device))
            
            epoch_reward += reward
            obs = next_obs
            
            # Update best model for this specific benchmark
            if info['proxy_score'] < best_proxies[bench_name]:
                best_proxies[bench_name] = info['proxy_score']
                model_dir = Path("models")
                model_dir.mkdir(exist_ok=True)
                torch.save(model.state_dict(), model_dir / f"gnn_placer_universal_best.pth")
                torch.save(model.state_dict(), model_dir / f"gnn_placer_{bench_name}_best.pth")
                print(f"  *** NEW BEST for {bench_name}: {info['proxy_score']:.4f} saved! ***")

            if done: break
            
        # PPO/REINFORCE Update
        if len(rewards) > 0:
            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + 0.99 * G
                returns.insert(0, G)
            
            returns = torch.stack(returns)
            
            loss = 0
            for lp, Gt in zip(log_probs, returns):
                loss -= (lp * Gt).mean() # log_prob * return
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                torch.save(model.state_dict(), "models/gnn_placer_universal_last.pth")
            
        print(f"Epoch {epoch}: Reward={epoch_reward:.4f}, Proxy={info['proxy_score']:.4f} (Best {bench_name}={best_proxies[bench_name]:.4f})")
        sys.stdout.flush()

if __name__ == "__main__":
    train()
