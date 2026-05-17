import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from typing import Optional, List, Tuple
import numpy as np

from graphplace.models import Benchmark

def plot_graph(
    benchmark: 'Benchmark',
    output_path: str = "graph_vis.png",
    max_edges: int = 10000,
    title: Optional[str] = None
):
    """
    Visualizes the graph where macros are nodes and nets are edges.
    
    Args:
        benchmark: The Benchmark object.
        output_path: Path to save the plot.
        max_edges: Maximum number of edges to draw (to avoid slowdown).
        title: Optional title for the plot.
    """
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 1. Setup Canvas
    ax.set_xlim(0, benchmark.canvas_width)
    ax.set_ylim(0, benchmark.canvas_height)
    ax.set_aspect('equal')
    
    canvas_rect = patches.Rectangle((0, 0), benchmark.canvas_width, benchmark.canvas_height, 
                                   linewidth=2, edgecolor='#333333', facecolor='#0a0a0a', zorder=0)
    ax.add_patch(canvas_rect)
    
    # 2. Draw Macros (Nodes)
    num_hard = benchmark.num_hard_macros
    added_labels = set()
    for i in range(benchmark.num_macros):
        w, h = benchmark.macro_sizes[i].tolist()
        cx, cy = benchmark.macro_positions[i].tolist()
        bl_x = cx - w/2
        bl_y = cy - h/2
        
        is_fixed = benchmark.macro_fixed[i].item()
        is_hard = i < num_hard
        
        if is_fixed:
            color = '#444444'; edge_color = '#666666'; alpha = 0.8; hatch = '///'
            label = 'Fixed Macro'
        elif is_hard:
            color = '#3498db'; edge_color = '#5dade2'; alpha = 0.7; hatch = None
            label = 'Hard Macro'
        else:
            color = '#9b59b6'; edge_color = '#af7ac5'; alpha = 0.5; hatch = None
            label = 'Soft Macro'
            
        rect = patches.Rectangle((bl_x, bl_y), w, h, 
                                linewidth=1, edgecolor=edge_color, 
                                facecolor=color, alpha=alpha, hatch=hatch, zorder=2)
        
        if label not in added_labels:
            rect.set_label(label)
            added_labels.add(label)
            
        ax.add_patch(rect)
        
    # 3. Draw Edges (Wire Connections)
    # Each net connects multiple nodes. We can draw lines from all nodes in a net to the net center
    # or use clique expansion (lines between all pairs). 
    # For visualization, drawing lines to the bounding box center or mean position is common.
    
    if hasattr(benchmark, 'net_nodes') and benchmark.net_nodes:
        edge_lines = []
        net_centers = []
        edge_count = 0
        
        for net_nodes in benchmark.net_nodes:
            if len(net_nodes) < 2:
                continue
            
            # Get positions of all nodes in this net
            pos = benchmark.macro_positions[net_nodes] # [N, 2]
            center = pos.mean(dim=0)
            net_centers.append(center.tolist())
            
            for p in pos:
                if edge_count >= max_edges:
                    break
                edge_lines.append([p.tolist(), center.tolist()])
                edge_count += 1
            
            if edge_count >= max_edges:
                print(f"Warning: Reached max_edges limit ({max_edges}). Some edges omitted.")
                break
        
        line_segments = LineCollection(edge_lines, colors='#f1c40f', linewidths=0.5, alpha=0.3, zorder=1)
        ax.add_collection(line_segments)
        
        # Draw net nodes
        if net_centers:
            nc = np.array(net_centers)
            ax.scatter(nc[:, 0], nc[:, 1], color='#e74c3c', s=20, zorder=3, label='Net Nodes')

    # 4. Finalize
    if title is None:
        title = f"Graph Visualization: {benchmark.name}\nNodes: {benchmark.num_macros}, Wires: {len(benchmark.net_nodes)}"
    
    ax.set_title(title, fontsize=16, fontweight='bold', color='white', pad=20)
    ax.set_xlabel("X (um)", fontsize=12, color='#888888')
    ax.set_ylabel("Y (um)", fontsize=12, color='#888888')
    
    # Add legend to explain colors
    ax.legend(loc='upper right', facecolor='#111111', edgecolor='#333333', labelcolor='white')
    
    plt.grid(color='#222222', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    print(f"Saving graph visualization to {output_path}...")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print("Done!")
