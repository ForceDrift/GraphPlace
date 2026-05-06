import torch
from pathlib import Path
import sys

# Add challenge path
challenge_path = Path("externals/macro-place-challenge-2026")
sys.path.append(str(challenge_path))

from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost

# Load benchmark
benchmark, plc = load_benchmark_from_dir("externals/macro-place-challenge-2026/external/MacroPlacement/Testcases/ICCAD04/ibm01")

# Load seed
pt_file = Path("data/generated/ibm01_gnn_seed.pt")
if pt_file.exists():
    data = torch.load(pt_file, weights_only=False)
    costs = compute_proxy_cost(data['macro_positions'], benchmark, plc)
    print(f"Proxy: {costs['proxy_cost']:.4f}")
    # The challenge evaluator doesn't always return num_overlaps in costs, but compute_proxy_cost checks it.
    # If the score is valid, it usually means overlaps are below threshold.
else:
    print("File not found")
