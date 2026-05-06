import torch
import torch.nn.functional as F
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Import components
from graphplace.core.gnn.gnn_placer import MacroPlacementGNN, prepare_features, build_bipartite_adj
from graphplace.core.legalizer import legalize
from graphplace.core.force_directed import visualize_placed_macros

# Challenge evaluator
challenge_path = Path("c:/Users/Roshan/code/GraphPlace/externals/macro-place-challenge-2026")
sys.path.append(str(challenge_path))
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost

def run_gnn_inference(seed_pt, out_pt, out_png, timesteps=100):
    print(f"Loading seed from {seed_pt}...")
    data = torch.load(seed_pt, weights_only=False)
    current_pos = data['macro_positions'].clone().float()
    sizes = data['macro_sizes'].float()
    fixed = data['macro_fixed']
    cw, ch = data['canvas_width'], data['canvas_height']
    
    # Load challenge environment for scoring
    benchmark_dir = challenge_path / "external/MacroPlacement/Testcases/ICCAD04/ibm01"
    benchmark, plc = load_benchmark_from_dir(benchmark_dir.as_posix())
    
    # Initialize GNN
    model = MacroPlacementGNN()
    # Note: Using random weights as we are demonstrating the inference loop
    model.eval()
    
    adj_m2n, adj_n2m = build_bipartite_adj(data['num_macros'], data['net_nodes'])
    
    scores = []
    
    # Initial score
    initial_costs = compute_proxy_cost(current_pos, benchmark, plc)
    initial_score = initial_costs['proxy_cost']
    print(f"Initial Proxy Score: {initial_score:.4f}")
    scores.append(initial_score)
    
    # High-leverage parameters for untangling
    step_size = 0.002 
    grid_size = 7 
    
    best_score = initial_score
    best_pos = current_pos.clone()
    
    print(f"Starting Hill-Climbing refinement for {timesteps} steps...", flush=True)
    
    for t in range(timesteps):
        # 1. Prepare features from current state
        m_feats, n_feats = prepare_features(data, current_pos)
        
        # 2. GNN Forward & Physics Force
        with torch.no_grad():
            logits, _ = model(m_feats, n_feats, adj_m2n, adj_n2m)
            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample()
            
            # Physics Force: Calculate center-of-gravity move (The secret to 1.30)
            net_centers = torch.matmul(adj_m2n, current_pos)
            macro_ideals = torch.matmul(adj_n2m, net_centers)
            ideal_dx = (macro_ideals[:, 0] - current_pos[:, 0])
            ideal_dy = (macro_ideals[:, 1] - current_pos[:, 1])
            
        # 3. Hybrid Proposal: 50% Physics-directed, 50% GNN-random
        noise_dx = (actions % grid_size - 3).float() * (cw * step_size)
        noise_dy = (actions // grid_size - 3).float() * (ch * step_size)
        
        dx = 0.5 * (ideal_dx * step_size * 5) + 0.5 * noise_dx
        dy = 0.5 * (ideal_dy * step_size * 5) + 0.5 * noise_dy
        
        proposed_pos = current_pos.clone()
        movable = ~fixed
        proposed_pos[movable, 0] += dx[movable]
        proposed_pos[movable, 1] += dy[movable]
        
        # 4. Legalize proposal
        proposed_pos = legalize(proposed_pos, sizes, fixed, cw, ch, max_iter=20)
        
        # 5. Evaluate and Accept if better
        costs = compute_proxy_cost(proposed_pos, benchmark, plc)
        new_score = costs['proxy_cost']
        
        if new_score < best_score:
            best_score = new_score
            current_pos = proposed_pos.clone()
            best_pos = current_pos.clone()
            print(f"Step {t+1}/{timesteps}: *** New Best Score: {best_score:.4f} ***", flush=True)
        elif (t+1) % 20 == 0:
            print(f"Step {t+1}/{timesteps}: Best = {best_score:.4f} (Try {new_score:.4f})", flush=True)
            
        scores.append(best_score)
                
    # Final high-quality legalization
    current_pos = legalize(best_pos, sizes, fixed, cw, ch, max_iter=1000)
    final_costs = compute_proxy_cost(current_pos, benchmark, plc)
    print(f"Final Proxy Score: {final_costs['proxy_cost']:.4f}")
    
    # Save and Visualize
    data['macro_positions'] = current_pos
    torch.save(data, out_pt)
    visualize_placed_macros(data, current_pos, ch, out_png)
    
    # Plot score progression
    plt.figure(figsize=(10, 5))
    plt.plot(scores, marker='o')
    plt.title("Proxy Score Progression (Untrained GNN Inference)")
    plt.xlabel("Evaluation Step (Every 5 steps)")
    plt.ylabel("Proxy Cost")
    plt.grid(True)
    plt.savefig(out_png.replace(".png", "_scores.png"))
    
    return scores

if __name__ == "__main__":
    run_gnn_inference(
        seed_pt="data/generated/ibm01_fd_spread.pt",
        out_pt="data/generated/ibm01_gnn_final.pt",
        out_png="ibm01_gnn_inference.png",
        timesteps=100
    )
    
