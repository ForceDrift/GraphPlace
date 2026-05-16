import sys
import torch
import os
import argparse

BASE_DIR = 'C:/Users/Roshan/code/GraphPlace'
CHALLENGE_DIR = os.path.join(BASE_DIR, 'externals/macro-place-challenge-2026')
sys.path.append(CHALLENGE_DIR)

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_proxy_cost
from macro_place.loader import load_benchmark_from_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=str, default='ibm01')
    parser.add_argument('--pt', type=str, required=True, help='Path to output .pt')
    args = parser.parse_args()

    pt_file = os.path.join(CHALLENGE_DIR, f'benchmarks/processed/public/{args.benchmark}.pt')
    if not os.path.exists(pt_file):
        print(f"Error: {pt_file} not found")
        return

    benchmark = Benchmark.load(pt_file)
    print(f"Loaded benchmark {args.benchmark} with {benchmark.num_macros} macros.")

    source_dir = os.path.join(BASE_DIR, f'externals/MacroPlacement/Testcases/ICCAD04/{args.benchmark}')
    _, plc = load_benchmark_from_dir(source_dir.replace('\\', '/'))

    placement = torch.load(args.pt)
    
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
