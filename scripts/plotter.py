import os
import sys
import torch
import csv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
from pathlib import Path
from typing import Optional

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from graphplace.core.models import Benchmark

def calculate_hpwl(benchmark: Benchmark) -> float:
    hpwl = torch.tensor(0.0)
    if not hasattr(benchmark, 'net_nodes') or not benchmark.net_nodes:
        return 0.0
        
    for net_nodes in benchmark.net_nodes:
        if len(net_nodes) < 2:
            continue
        positions = benchmark.macro_positions[net_nodes]
        
        min_x = torch.min(positions[:, 0])
        max_x = torch.max(positions[:, 0])
        min_y = torch.min(positions[:, 1])
        max_y = torch.max(positions[:, 1])
        
        hpwl = hpwl + (max_x - min_x) + (max_y - min_y)
        
    return hpwl.item()

def plot_placement(csv_path: str, benchmark_path: Optional[str] = None, output_path: str = "placement_plot.png"):
    """
    Plots the placement from a CSV file using dimensions from a benchmark file.
    """
    print(f"Reading placement from {csv_path}...")
    
    nodes_data = []
    with open(csv_path, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            nodes_data.append({
                'node_name': row['node_name'],
                'x': float(row['x']),
                'y': float(row['y'])
            })
    if benchmark_path is None:
        csv_name = os.path.basename(csv_path)
        base_name = csv_name.split("_sample_")[0] if "_sample_" in csv_name else csv_name.split("_placement")[0]
        benchmark_path = os.path.join("data", "processed", "public", f"{base_name}.pt")
        
    if not os.path.exists(benchmark_path):
        print(f"Warning: Benchmark file {benchmark_path} not found. Searching...")
        found = False
        for p in Path("data").rglob(f"{os.path.basename(benchmark_path)}"):
            benchmark_path = str(p)
            found = True
            break
        if not found:
            raise FileNotFoundError(f"Could not find benchmark file for {csv_path}")

    print(f"Loading benchmark dimensions from {benchmark_path}...")
    benchmark = Benchmark.load(benchmark_path)
    
    name_to_idx = {name: i for i, name in enumerate(benchmark.macro_names)}
    
    updated_positions = benchmark.macro_positions.clone()
    for row in nodes_data:
        node_name = row['node_name']
        if node_name in name_to_idx:
            idx = name_to_idx[node_name]
            updated_positions[idx, 0] = row['x']
            updated_positions[idx, 1] = row['y']
    
    benchmark.macro_positions = updated_positions
    hpwl = calculate_hpwl(benchmark)
    print(f"Calculated HPWL: {hpwl:.2f} um")
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 10))
    
    ax.set_xlim(0, benchmark.canvas_width)
    ax.set_ylim(0, benchmark.canvas_height)
    ax.set_aspect('equal')
    
    canvas_rect = patches.Rectangle((0, 0), benchmark.canvas_width, benchmark.canvas_height, 
                                   linewidth=2, edgecolor='#333333', facecolor='#0a0a0a', zorder=0)
    ax.add_patch(canvas_rect)
    num_hard = benchmark.num_hard_macros
    for i in range(benchmark.num_macros):
        w, h = benchmark.macro_sizes[i]
        cx, cy = benchmark.macro_positions[i]
        bl_x = cx - w/2
        bl_y = cy - h/2
        
        is_fixed = benchmark.macro_fixed[i].item()
        is_hard = i < num_hard
        
        if is_fixed:
            color = '#444444'; edge_color = '#666666'; alpha = 0.8; hatch = '///'
        elif is_hard:
            color = '#3498db'; edge_color = '#5dade2'; alpha = 0.7; hatch = None
        else:
            color = '#9b59b6'; edge_color = '#af7ac5'; alpha = 0.5; hatch = None
            
        rect = patches.Rectangle((bl_x, bl_y), w, h, 
                                linewidth=1, edgecolor=edge_color, 
                                facecolor=color, alpha=alpha, hatch=hatch)
        ax.add_patch(rect)

    ax.set_title(f"Placement: {benchmark.name}\nHPWL: {hpwl:,.2f} \u03BCm", 
                 fontsize=16, fontweight='bold', color='white', pad=20)
    ax.set_xlabel("X (\u03BCm)", fontsize=12, color='#888888')
    ax.set_ylabel("Y (\u03BCm)", fontsize=12, color='#888888')
    
    legend_elements = [
        patches.Patch(facecolor='#444444', edgecolor='#666666', hatch='///', label='Fixed Macro'),
        patches.Patch(facecolor='#3498db', edgecolor='#5dade2', alpha=0.7, label='Hard Macro'),
        patches.Patch(facecolor='#9b59b6', edgecolor='#af7ac5', alpha=0.5, label='Soft Macro')
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True, facecolor='#111111', edgecolor='#333333')
    
    plt.grid(color='#222222', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    print(f"Saving plot to {output_path}...")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize placement CSV files.")
    parser.add_argument("csv", type=str, help="Path to the placement CSV file")
    parser.add_argument("--benchmark", type=str, default=None, help="Path to the benchmark .pt file (optional)")
    parser.add_argument("--out", type=str, default="placement_plot.png", help="Output image path")
    
    args = parser.parse_args()
    plot_placement(args.csv, args.benchmark, args.out)
