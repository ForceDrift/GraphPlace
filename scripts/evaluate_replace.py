import sys
import torch
import os
import argparse
import numpy as np

# Add macro_place to path
BASE_DIR = '/Users/roshaniruku/code/GraphPlace'
CHALLENGE_DIR = os.path.join(BASE_DIR, 'externals/macro-place-challenge-2026')
sys.path.append(CHALLENGE_DIR)

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_proxy_cost
from macro_place.loader import load_benchmark_from_dir

def parse_pl(pl_file):
    positions = {}
    with open(pl_file, 'r') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '' or line.startswith('UCLA'):
                continue
            parts = line.split()
            if len(parts) >= 3:
                node_name = parts[0]
                try:
                    x = float(parts[1])
                    y = float(parts[2])
                    positions[node_name] = (x, y)
                except ValueError:
                    continue
    return positions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=str, default='ibm01')
    parser.add_argument('--pl', type=str, required=True, help='Path to RePlAce .pl output')
    args = parser.parse_args()

    # 1. Load benchmark from challenge repo (processed .pt version)
    pt_file = os.path.join(CHALLENGE_DIR, f'benchmarks/processed/public/{args.benchmark}.pt')
    if not os.path.exists(pt_file):
        print(f"Error: {pt_file} not found")
        return

    benchmark = Benchmark.load(pt_file)
    print(f"Loaded benchmark {args.benchmark} with {benchmark.num_macros} macros.")

    # 2. Load PlacementCost object (needed for some evaluators)
    # The evaluation scripts in reach-challenge usually load it from the source dir
    source_dir = os.path.join(CHALLENGE_DIR, f'external/MacroPlacement/Testcases/ICCAD04/{args.benchmark}')
    _, plc = load_benchmark_from_dir(source_dir)

    # 3. Parse RePlAce .pl file
    replace_pos = parse_pl(args.pl)
    
    # 4. Prepare placement tensor (centers)
    # RePlAce .pl contains bottom-left coordinates.
    # benchmark.macro_positions is [num_macros, 2]
    placement = benchmark.macro_positions.clone()
    
    found_count = 0
    for i, name in enumerate(benchmark.macro_names):
        if name in replace_pos:
            bl_x, bl_y = replace_pos[name]
            w, h = benchmark.macro_sizes[i].tolist()
            # Convert bottom-left to center
            placement[i, 0] = bl_x + w / 2.0
            placement[i, 1] = bl_y + h / 2.0
            found_count += 1
    
    print(f"Updated positions for {found_count}/{benchmark.num_macros} macros.")

    # 5. Evaluate
    results = compute_proxy_cost(placement, benchmark, plc)
    
    print("\n" + "="*40)
    print(f"   Evaluation Results for {args.benchmark}")
    print("="*40)
    print(f"Proxy Cost:      {results['proxy_cost']:.4f}")
    print(f"Wirelength:      {results['wirelength_cost']:.4f}")
    print(f"Density Cost:    {results['density_cost']:.4f}")
    print(f"Congestion Cost: {results['congestion_cost']:.4f}")
    print(f"Overlap Count:   {results['overlap_count']}")
    if results['overlap_count'] > 0:
        print(f"Total Overlap:   {results['total_overlap_area']:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()
