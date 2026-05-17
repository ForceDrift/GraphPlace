import os
import sys
import torch
import torch.nn as nn
import argparse
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from graphplace.rl.placement_env import PlacementEnv
from graphplace.gnn_placer import PlaceGNN
from torch.distributions import Normal

def run_eval(bench_name, model_path, device="cuda"):
    print(f"\n--- EVALUATING MODEL: {model_path} ---")
    
    # 1. Setup Env and Model
    env = PlacementEnv(benchmark_name=bench_name)
    model = PlaceGNN()
    
    # 2. Check for warm-start
    warm_start_path = Path("output") / bench_name / f"{bench_name}_legalized.pt"
    warm_start_pos = torch.load(warm_start_path) if warm_start_path.exists() else None
    
    # 3. Load the specific version
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded: {model_path}")
    else:
        print(f"Error: {model_path} does not exist!")
        return

    model.to(device)
    model.eval()

    # 4. Run 1 sequence
    obs = env.reset(options={'warm_start_pos': warm_start_pos} if warm_start_pos is not None else None)
    initial_score = env.last_score
    print(f"Start Score: {initial_score:.4f}")
    
    with torch.no_grad():
        for i in range(50):
            obs_tensor = torch.from_numpy(obs).float().to(device).unsqueeze(0)
            mu, _ = model(obs_tensor)
            action = mu.squeeze(0).cpu().numpy()
            obs, reward, done, info = env.step(action, legalize=False)
            if done: break
            
    print(f"End Score:   {info['proxy_score']:.4f}")
    improvement = initial_score - info['proxy_score']
    print(f"Net Gain:    {improvement:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--bench", type=str, default="ibm01")
    args = parser.parse_args()
    
    run_eval(args.bench, args.model)
