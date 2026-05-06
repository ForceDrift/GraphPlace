"""
Aggressive Simulated Annealing + Physics-Guided GNN Refinement.

Strategy:
1. Start from the force-directed seed
2. Use SA to escape local minima (accept worse moves early on)
3. Physics forces pull macros toward their connectivity center
4. Gradually cool down to pure hill-climbing
5. Multi-restart: try several seeds and keep the best
"""
import torch
import torch.nn.functional as F
import math
import sys
from pathlib import Path
import matplotlib.pyplot as plt

from graphplace.core.gnn.gnn_placer import MacroPlacementGNN, prepare_features, build_bipartite_adj
from graphplace.core.legalizer import legalize
from graphplace.core.force_directed import visualize_placed_macros

challenge_path = Path("c:/Users/Roshan/code/GraphPlace/externals/macro-place-challenge-2026")
sys.path.append(str(challenge_path))
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost


def run_sa_refinement(seed_pt, out_pt, out_png, benchmark_name="ibm01",
                      timesteps=500, num_restarts=3):
    """Simulated Annealing refinement with physics-guided moves."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print(f"Loading seed from {seed_pt}...")
    data = torch.load(seed_pt, weights_only=False)
    sizes = data['macro_sizes'].float().to(device)
    fixed = data['macro_fixed'].to(device)
    cw, ch = data['canvas_width'], data['canvas_height']
    
    benchmark_dir = challenge_path / f"external/MacroPlacement/Testcases/ICCAD04/{benchmark_name}"
    benchmark, plc = load_benchmark_from_dir(benchmark_dir.as_posix())
    
    adj_m2n, adj_n2m = build_bipartite_adj(data['num_macros'], data['net_nodes'])
    adj_m2n, adj_n2m = adj_m2n.to(device), adj_n2m.to(device)
    movable = ~fixed
    num_macros = data['num_macros']
    num_hard = data.get('num_hard_macros', num_macros)
    
    global_best_score = float('inf')
    global_best_pos = None
    
    for restart in range(num_restarts):
        print(f"\n{'='*60}")
        print(f"RESTART {restart+1}/{num_restarts}")
        print(f"{'='*60}")
        
        # Initialize GNN with fresh random weights each restart
        model = MacroPlacementGNN().to(device)
        model.eval()
        
        current_pos = data['macro_positions'].clone().float().to(device)
        
        # Initial score (Evaluator needs CPU)
        initial_costs = compute_proxy_cost(current_pos.cpu(), benchmark, plc)
        current_score = initial_costs['proxy_cost']
        print(f"Initial Score: {current_score:.4f} "
              f"(wl={initial_costs['wirelength_cost']:.3f} "
              f"den={initial_costs['density_cost']:.3f} "
              f"cong={initial_costs['congestion_cost']:.3f})")
        
        best_score = current_score
        best_pos = current_pos.clone()
        
        # SA parameters
        T_start = 2.0   # Initial temperature (accept moves up to +2.0 in score)
        T_end = 0.001    # Final temperature (essentially greedy)
        
        # Adaptive step size: start large, shrink
        step_start = 0.01
        step_end = 0.0005
        
        for t in range(timesteps):
            progress = t / max(timesteps - 1, 1)
            
            # Exponential temperature decay
            T = T_start * (T_end / T_start) ** progress
            
            # Adaptive step size
            step_size = step_start * (step_end / step_start) ** progress
            
            # Physics Force: center-of-gravity pull
            net_centers = torch.matmul(adj_m2n, current_pos)
            macro_ideals = torch.matmul(adj_n2m, net_centers)
            ideal_dx = (macro_ideals[:, 0] - current_pos[:, 0])
            ideal_dy = (macro_ideals[:, 1] - current_pos[:, 1])
            
            # GNN noise component
            m_feats, n_feats = prepare_features(data, current_pos)
            with torch.no_grad():
                logits, _ = model(m_feats, n_feats, adj_m2n, adj_n2m)
                dist = torch.distributions.Categorical(logits=logits)
                actions = dist.sample()
            
            noise_dx = (actions % 7 - 3).float() * (cw * step_size)
            noise_dy = (actions // 7 - 3).float() * (ch * step_size)
            
            # Physics weight increases over time (more greedy later)
            physics_weight = 0.3 + 0.5 * progress  # 0.3 -> 0.8
            noise_weight = 1.0 - physics_weight
            
            dx = physics_weight * (ideal_dx * step_size * 5) + noise_weight * noise_dx
            dy = physics_weight * (ideal_dy * step_size * 5) + noise_weight * noise_dy
            
            # Move a random subset (more macros early, fewer later)
            move_ratio = 0.3 * (1 - progress) + 0.05  # 0.35 -> 0.05
            num_to_move = max(1, int(num_macros * move_ratio))
            move_indices = torch.randperm(num_macros)[:num_to_move]
            
            proposed_pos = current_pos.clone()
            for idx in move_indices:
                if movable[idx]:
                    proposed_pos[idx, 0] += dx[idx]
                    proposed_pos[idx, 1] += dy[idx]
            
            # Clamp to canvas before legalization
            proposed_pos[:, 0] = proposed_pos[:, 0].clamp(
                sizes[:, 0] / 2, cw - sizes[:, 0] / 2)
            proposed_pos[:, 1] = proposed_pos[:, 1].clamp(
                sizes[:, 1] / 2, ch - sizes[:, 1] / 2)
            
            # 4. Fast Vectorized Legalization (skip_greedy=True)
            proposed_pos = legalize(proposed_pos, sizes, fixed, cw, ch,
                                    hard_only=True, num_hard=num_hard,
                                    max_iter=100, skip_greedy=True)
            
            # Every 25 steps, do a full "hard" legalization to keep the score accurate
            if (t + 1) % 25 == 0:
                proposed_pos = legalize(proposed_pos, sizes, fixed, cw, ch,
                                        hard_only=True, num_hard=num_hard,
                                        max_iter=500, skip_greedy=False)
            
            # 5. Evaluate (Evaluator needs CPU)
            costs = compute_proxy_cost(proposed_pos.cpu(), benchmark, plc)
            new_score = costs['proxy_cost']
            
            # Simulated Annealing acceptance
            delta = new_score - current_score
            if delta < 0:
                # Always accept improvements
                accept = True
            elif T > 0:
                # Accept worse moves with probability exp(-delta/T)
                accept_prob = math.exp(-delta / T)
                accept = torch.rand(1).item() < accept_prob
            else:
                accept = False
            
            if accept:
                current_score = new_score
                current_pos = proposed_pos.clone()
                
                if new_score < best_score:
                    best_score = new_score
                    best_pos = current_pos.clone()
                    print(f"  Step {t+1}/{timesteps} [T={T:.4f}]: "
                          f"*** BEST {best_score:.4f} *** "
                          f"(wl={costs['wirelength_cost']:.3f} "
                          f"den={costs['density_cost']:.3f} "
                          f"cong={costs['congestion_cost']:.3f})", flush=True)
            
            if (t+1) % 10 == 0 and not accept:
                print(f"  Step {t+1}/{timesteps} [T={T:.4f}]: "
                      f"Best={best_score:.4f} Current={current_score:.4f}", flush=True)
        
        # Final legalization on this restart's best
        final_pos = legalize(best_pos, sizes, fixed, cw, ch,
                             hard_only=True, num_hard=num_hard,
                             max_iter=1000, skip_greedy=False)
        final_costs = compute_proxy_cost(final_pos.cpu(), benchmark, plc)
        print(f"Restart {restart+1} Final: {final_costs['proxy_cost']:.4f} "
              f"(wl={final_costs['wirelength_cost']:.3f} "
              f"den={final_costs['density_cost']:.3f} "
              f"cong={final_costs['congestion_cost']:.3f})")
        
        if final_costs['proxy_cost'] < global_best_score:
            global_best_score = final_costs['proxy_cost']
            global_best_pos = final_pos.cpu().clone()
            print(f"  >>> NEW GLOBAL BEST: {global_best_score:.4f} <<<")
    
    # Save global best
    print(f"\n{'='*60}")
    print(f"FINAL RESULT: {global_best_score:.4f}")
    print(f"{'='*60}")
    
    data['macro_positions'] = global_best_pos
    torch.save(data, out_pt)
    visualize_placed_macros(data, global_best_pos, ch, out_png)
    

if __name__ == "__main__":
    run_sa_refinement(
        seed_pt="data/generated/ibm01_fd_spread.pt",
        out_pt="data/generated/ibm01_gnn_final.pt",
        out_png="ibm01_sa_optimized.png",
        benchmark_name="ibm01",
        timesteps=500,
        num_restarts=3
    )
