import os
import sys
import torch
import random
import argparse
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from graphplace.models import Benchmark
from graphplace.legalizer import Legalizer

def generate_samples(input_path: str, num_samples: int, output_dir: str):
    """
    Generates random placement samples from a base benchmark and legalizes them.
    """
    print(f"Loading base benchmark: {input_path}")
    base_benchmark = Benchmark.load(input_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    canvas_w = base_benchmark.canvas_width
    canvas_h = base_benchmark.canvas_height
    
    for i in range(num_samples):
        sample = Benchmark.load(input_path)
        
        movable_mask = sample.get_movable_mask()
        new_x = torch.rand(sample.num_macros) * canvas_w
        new_y = torch.rand(sample.num_macros) * canvas_h
        
        sample.macro_positions[movable_mask, 0] = new_x[movable_mask]
        sample.macro_positions[movable_mask, 1] = new_y[movable_mask]
        
        legalizer = Legalizer(sample)
        is_legal = legalizer.legalize()
        
        csv_filename = f"{sample.name}_sample_{i}_placement.csv"
        csv_path = os.path.join(output_dir, csv_filename)
        with open(csv_path, "w") as f:
            f.write("node_name,x,y\n")
            for j in range(sample.num_macros):
                f.write(f"{sample.macro_names[j]},{sample.macro_positions[j, 0]:.4f},{sample.macro_positions[j, 1]:.4f}\n")
        
        status = "Legalized" if is_legal else "Partially Legalized"
        print(f"Generated sample {i}: {csv_filename} ({status})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate random placement samples from .pt benchmarks.")
    parser.add_argument("--input", type=str, default="data/processed/public/ariane133_ng45.pt", help="Path to input .pt benchmark")
    parser.add_argument("--num", type=int, default=5, help="Number of samples to generate")
    parser.add_argument("--out", type=str, default="data/samples", help="Output directory")
    
    args = parser.parse_args()
    
    generate_samples(args.input, args.num, args.out)
