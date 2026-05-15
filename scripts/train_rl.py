import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from tqdm import tqdm
import argparse

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from graphplace.rl.placement_env import PlacementEnv
from graphplace.models.gnn_placer import PlaceGNN
from graphplace.graph.pyg_converter import to_hetero_data

class PPOAgent:
    def __init__(self, model, lr=3e-4, gamma=0.99, eps_clip=0.2):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.MseLoss = nn.MSELoss()

    def select_action(self, state_data):
        # state_data is HeteroData
        _, offsets = self.model(state_data)
        # For RL, we use the offsets as the mean of a Normal distribution
        # In this simplistic version, we'll just use the offsets directly for rollout
        return offsets.detach()

    def update(self, rollouts):
        # Standard PPO update logic
        # ...
        pass

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", type=str, default="ibm01")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps_per_epoch", type=int, default=50)
    parser.add_argument("--device", type=str, default=None, help="Device to use (cpu, cuda). Auto-detects if not set.")
    args = parser.parse_args()

    env = PlacementEnv(benchmark_name=args.bench)
    model = PlaceGNN()
    
    # Device setup
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    
    # We need the graph structure for the GNN
    from graphplace.core.models import Benchmark
    from graphplace.graph.pyg_converter import parse_netlist_pb
    bench_pt = f"data/processed/public/{args.bench}.pt"
    benchmark = Benchmark.load(bench_pt)
    netlist_path = f"externals/MacroPlacement/Testcases/ICCAD04/{args.bench}/netlist.pb.txt"
    _, net_nodes, _ = parse_netlist_pb(netlist_path)
    graph_data = to_hetero_data(benchmark, net_nodes=net_nodes).to(device)

    # Load warm start if available to refine RePlAce's placement
    warm_start_pos = None
    warm_start_path = Path("output") / args.bench / f"{args.bench}_legalized.pt"
    if warm_start_path.exists():
        warm_start_pos = torch.load(warm_start_path)
        print(f"Loaded Warm Start placement from {warm_start_path}")

    # Training Loop
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    global_best_proxy = float('inf')

    for epoch in range(args.epochs):
        options = {'warm_start_pos': warm_start_pos} if warm_start_pos is not None else None
        obs = env.reset(options=options)
        epoch_reward = 0
        
        # Collect Trajectory
        log_probs = []
        rewards = []
        
        for step in range(args.steps_per_epoch):
            # Create a clone for this step to avoid in-place PyTorch errors across rollout steps
            step_graph_data = graph_data.clone()

            # Update step_graph_data with current positions from environment (Non-inplace)
            new_pos = torch.tensor(obs[:, :2], device=device, dtype=torch.float32)
            step_graph_data['macro'].x = torch.cat([new_pos, step_graph_data['macro'].x[:, 2:]], dim=1)
            
            # Forward: Get Mean displacement from GNN
            _, mu = model(step_graph_data)
            
            # Policy: Normal distribution for exploration
            # Use small fixed std for micro-nudges
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
            
            # Update best model if improvement found
            if info['proxy_score'] < global_best_proxy:
                global_best_proxy = info['proxy_score']
                model_dir = Path("models")
                model_dir.mkdir(exist_ok=True)
                torch.save(model.state_dict(), model_dir / f"gnn_placer_{args.bench}_best.pth")
                print(f"  *** NEW BEST: {global_best_proxy:.4f} saved! ***")

            if done: break
            
        # PPO/REINFORCE Update
        if len(rewards) > 0:
            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + 0.99 * G
                returns.insert(0, G)
            
            returns = torch.stack(returns)
            # Whiten returns
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            loss = 0
            for lp, Gt in zip(log_probs, returns):
                loss -= (lp * Gt).mean() # log_prob * return
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Save periodic model
            if epoch % 10 == 0:
                torch.save(model.state_dict(), f"models/gnn_placer_{args.bench}_last.pth")
            
        print(f"Epoch {epoch}: Reward={epoch_reward:.4f}, Proxy={info['proxy_score']:.4f} (Best={global_best_proxy:.4f})")
        sys.stdout.flush()

if __name__ == "__main__":
    train()
