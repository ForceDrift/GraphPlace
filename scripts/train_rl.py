import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from tqdm import tqdm
import argparse
import os

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
    args = parser.parse_args()

    env = PlacementEnv(benchmark_name=args.bench)
    model = PlaceGNN()
    
    # Pre-load graph structure
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # We need the graph structure for the GNN
    from graphplace.core.models import Benchmark
    from graphplace.graph.pyg_converter import parse_netlist_pb
    bench_pt = f"data/processed/public/{args.bench}.pt"
    benchmark = Benchmark.load(bench_pt)
    netlist_path = f"externals/MacroPlacement/Testcases/ICCAD04/{args.bench}/netlist.pb.txt"
    _, net_nodes, _ = parse_netlist_pb(netlist_path)
    graph_data = to_hetero_data(benchmark, net_nodes=net_nodes).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(args.epochs):
        obs = env.reset() # Gym returns only obs
        epoch_reward = 0
        
        for step in range(args.steps_per_epoch):
            # Update graph_data with current positions from environment
            graph_data['macro'].x[:, :2] = torch.tensor(obs[:, :2], device=device)
            
            # Forward
            _, offsets = model(graph_data)
            
            # Take action in env
            action = offsets.cpu().detach().numpy()
            next_obs, reward, done, info = env.step(action) # Gym returns 4 values
            
            # Simple PG Loss (demo implementation)
            # In a real PPO, we'd store trajectories and use advantage estimates
            loss = -torch.mean(offsets * reward) # REINFORCE-style dummy
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_reward += reward
            obs = next_obs
            
            if done: break
            
        print(f"Epoch {epoch}: Reward={epoch_reward:.4f}, Final Proxy={info['proxy_score']:.4f}")

if __name__ == "__main__":
    train()
