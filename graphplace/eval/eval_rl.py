import os
import torch
import torch.nn as nn
import argparse
from pathlib import Path
from graphplace.rl.placement_env import PlacementEnv
from graphplace.model.gnn import PlaceGNN
from scripts.legalize_challenge import greedy_refine

def run_inference(bench_name, model_path="best_model.pt", steps=50, device="cuda"):
    print(f"Starting inference on {bench_name} using {model_path}...")
    
    # 1. Setup Environment
    env = PlacementEnv(benchmark_name=bench_name)
    
    # 2. Check for warm-start placement
    warm_start_path = Path("output") / bench_name / f"{bench_name}_legalized.pt"
    warm_start_pos = None
    if warm_start_path.exists():
        warm_start_pos = torch.load(warm_start_path)
    
    # 3. Load Model
    model = PlaceGNN()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded trained model: {model_path}")
    else:
        print(f"ERROR: Model file {model_path} not found!")
        return

    device = torch.device(device)
    model.to(device)
    model.eval()

    # 4. Starting Placement
    obs = env.reset(options={'warm_start_pos': warm_start_pos} if warm_start_pos is not None else None)
    initial_score = env.last_score
    print(f"Initial Score (from RePlAce): {initial_score:.4f}")

    # 5. Run GNN Steps
    with torch.no_grad():
        for i in range(steps):
            # Convert observation to tensor
            obs_tensor = torch.from_numpy(obs).float().to(device).unsqueeze(0)
            
            # Use model to get actions (displacements)
            # In this project, PlaceGNN outputs mu, sigma for policy
            # We just take the mean (mu) for inference
            mu, sigma = model(obs_tensor)
            action = mu.squeeze(0).cpu().numpy()
            
            # Take step (without legalization during steps for speed)
            obs, reward, done, info = env.step(action, legalize=False)
            
            if (i+1) % 10 == 0:
                print(f"  Step {i+1}/{steps}: Current Proxy = {info['proxy_score']:.4f}")

    # 6. FINAL LEGALIZATION (The slow, perfect pass)
    print("Performing final legalization (Greedy Refine)...")
    final_pos = greedy_refine(env.current_pos.clone(), env.mp_benchmark)
    
    # 7. Final Scoring
    # We use the environment's full score (non-fast) for the final result
    final_score = env._get_score(final_pos, fast=False)
    
    print("\n" + "="*40)
    print(f"INFERENCE COMPLETE for {bench_name}")
    print(f"Baseline Score: {initial_score:.4f}")
    print(f"FINAL GNN SCORE: {final_score:.4f}")
    print(f"IMPROVEMENT: {initial_score - final_score:.4f}")
    print("="*40)

    # 8. Save Result
    output_dir = Path("output") / bench_name
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{bench_name}_gnn_final.pt"
    torch.save(final_pos, out_path)
    print(f"Final placement saved to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", type=str, default="ibm01")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--model", type=str, default="best_model.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    run_inference(args.bench, args.model, args.steps, args.device)
