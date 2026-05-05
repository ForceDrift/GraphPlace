import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import os
from pathlib import Path

# Import our GNN components
from graphplace.core.gnn_placer import MacroPlacementGNN, prepare_features, build_bipartite_adj
from graphplace.core.legalizer import legalize
# Import Challenge Evaluator
import sys
challenge_path = Path("c:/Users/Roshan/code/GraphPlace/externals/macro-place-challenge-2026")
sys.path.append(str(challenge_path))
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost

class PlacementEnv:
    def __init__(self, benchmark_name="ibm01"):
        self.benchmark_name = benchmark_name
        testcase_root = challenge_path / "external/MacroPlacement/Testcases/ICCAD04"
        benchmark_dir = testcase_root / benchmark_name
        
        print(f"Loading environment for {benchmark_name}...")
        self.benchmark, self.plc = load_benchmark_from_dir(benchmark_dir.as_posix())
        
        # Internal state
        self.data = torch.load(f"data/processed/public/{benchmark_name}.pt", weights_only=False)
        self.current_pos = self.data['macro_positions'].clone().float()
        self.sizes = self.data['macro_sizes'].float()
        self.fixed = self.data['macro_fixed']
        self.cw = self.data['canvas_width']
        self.ch = self.data['canvas_height']
        
        # Initial score
        initial_costs = compute_proxy_cost(self.current_pos, self.benchmark, self.plc)
        self.last_score = initial_costs['proxy_cost']
        print(f"  Initial Proxy Score: {self.last_score:.4f}")

    def reset(self):
        self.current_pos = self.data['macro_positions'].clone().float()
        initial_costs = compute_proxy_cost(self.current_pos, self.benchmark, self.plc)
        self.last_score = initial_costs['proxy_cost']
        return self.get_obs()

    def get_obs(self):
        m_feats, n_feats = prepare_features(self.data, self.current_pos)
        adj_m2n, adj_n2m = build_bipartite_adj(self.data['num_macros'], self.data['net_nodes'])
        return m_feats, n_feats, adj_m2n, adj_n2m

    def step(self, actions, step_size=0.01):
        """
        Apply local grid actions to each macro.
        actions: tensor of indices [num_macros]
        """
        grid_size = 7
        dx = (actions % grid_size - (grid_size // 2)).float() * (self.cw * step_size)
        dy = (actions // grid_size - (grid_size // 2)).float() * (self.ch * step_size)
        
        # Update positions
        new_pos = self.current_pos.clone()
        # Only move non-fixed macros
        movable = ~self.fixed
        new_pos[movable, 0] += dx[movable]
        new_pos[movable, 1] += dy[movable]
        
        # Legalize (Step 4 requirement)
        # print("  Legalizing...")
        legal_pos = legalize(new_pos, self.sizes, self.fixed, self.cw, self.ch, max_iter=100)
        self.current_pos = legal_pos
        
        # Evaluate
        costs = compute_proxy_cost(self.current_pos, self.benchmark, self.plc)
        new_score = costs['proxy_cost']
        
        # Reward = Improvement (Step 4 requirement)
        reward = self.last_score - new_score
        self.last_score = new_score
        
        return self.get_obs(), reward, False, costs

def train_one_episode(env, model, optimizer, steps=10):
    model.train()
    obs = env.reset()
    
    trajectory = []
    total_reward = 0
    
    for t in range(steps):
        m_feats, n_feats, adj_m2n, adj_n2m = obs
        
        # GNN forward pass
        logits, value = model(m_feats, n_feats, adj_m2n, adj_n2m)
        
        # Sample actions for each macro
        dist = Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        
        # Step environment
        next_obs, reward, done, info = env.step(actions)
        
        # Store transition
        trajectory.append({
            'log_probs': log_probs,
            'value': value,
            'reward': reward
        })
        
        total_reward += reward
        obs = next_obs
        
    # Update Model (REINFORCE / Actor-Critic simplified)
    returns = []
    R = 0
    for transition in reversed(trajectory):
        R = transition['reward'] + 0.9 * R
        returns.insert(0, R)
    
    returns = torch.tensor(returns)
    # Normalize returns for stability
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    policy_loss = 0
    value_loss = 0
    
    for transition, R in zip(trajectory, returns):
        advantage = R - transition['value'].item()
        policy_loss -= (transition['log_probs'] * advantage).mean()
        value_loss += F.mse_loss(transition['value'], torch.tensor([[R]]))
        
    loss = policy_loss + 0.5 * value_loss
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return total_reward, env.last_score

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    env = PlacementEnv("ibm01")
    model = MacroPlacementGNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    print("Starting Training Loop...")
    for i in range(20):
        reward, final_score = train_one_episode(env, model, optimizer, steps=5)
        print(f"Episode {i+1}: Total Reward={reward:.4f}, Final Proxy={final_score:.4f}")
    
    # Save the trained model
    torch.save(model.state_dict(), "data/generated/gnn_rl_model.pth")
    print("Training complete. Model saved.")
