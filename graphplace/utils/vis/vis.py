import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from typing import Optional, List, Tuple
import numpy as np

from typing import Optional, List, Tuple, Any
import numpy as np

class BenchmarkPlotter:
    @staticmethod
    def plot_benchmark(
        benchmark: Any,
        output_path: str = "placement_vis.png",
        draw_edges: bool = True,
        max_edges: int = 10000,
        title: Optional[str] = None
    ):
        """
        Visualizes the benchmark placement.
        """
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # We will set limits at the end after scanning all macro positions.
        ax.set_aspect('equal')
        
        canvas_rect = patches.Rectangle((0, 0), benchmark.canvas_width, benchmark.canvas_height, 
                                       linewidth=2, edgecolor='#333333', facecolor='#0a0a0a', zorder=0)
        ax.add_patch(canvas_rect)
        
        # 2. Draw Macros
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
            
        # 3. Draw Edges
        if draw_edges and hasattr(benchmark, 'net_nodes') and benchmark.net_nodes:
            edge_lines = []
            edge_count = 0
            for net_nodes in benchmark.net_nodes:
                if len(net_nodes) < 2:
                    continue
                pos = benchmark.macro_positions[net_nodes]
                center = pos.mean(dim=0)
                for p in pos:
                    if edge_count >= max_edges:
                        break
                    edge_lines.append([p.tolist(), center.tolist()])
                    edge_count += 1
                if edge_count >= max_edges:
                    break
            
            line_segments = LineCollection(edge_lines, colors='#f1c40f', linewidths=0.5, alpha=0.2, zorder=1)
            ax.add_collection(line_segments)

        # 4. Compute and Set Limits
        try:
            pos = benchmark.macro_positions
            sizes = benchmark.macro_sizes
            min_xs = (pos[:, 0] - sizes[:, 0]/2).min().item()
            max_xs = (pos[:, 0] + sizes[:, 0]/2).max().item()
            min_ys = (pos[:, 1] - sizes[:, 1]/2).min().item()
            max_ys = (pos[:, 1] + sizes[:, 1]/2).max().item()
            
            # Include canvas bounds
            min_x = min(0, min_xs)
            max_x = max(benchmark.canvas_width, max_xs)
            min_y = min(0, min_ys)
            max_y = max(benchmark.canvas_height, max_ys)
            
            # Add 5% margin
            margin_x = (max_x - min_x) * 0.05
            margin_y = (max_y - min_y) * 0.05
            
            ax.set_xlim(min_x - margin_x, max_x + margin_x)
            ax.set_ylim(min_y - margin_y, max_y + margin_y)
        except Exception as e:
            # Fallback
            ax.set_xlim(0, benchmark.canvas_width)
            ax.set_ylim(0, benchmark.canvas_height)

        # 5. Finalize
        if title is None:
            title = f"Placement: {benchmark.name}\nMacros: {benchmark.num_macros}, Nets: {benchmark.num_nets}"
        
        ax.set_title(title, fontsize=16, fontweight='bold', color='white', pad=20)
        ax.set_xlabel("X (um)", fontsize=12, color='#888888')
        ax.set_ylabel("Y (um)", fontsize=12, color='#888888')
        ax.legend(loc='upper right', facecolor='#111111', edgecolor='#333333', labelcolor='white')
        
        plt.grid(color='#222222', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        
        print(f"Saving visualization to {output_path}...")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print("Done!")

def plot_graph(benchmark: Any, output_path: str = "graph_vis.png", **kwargs):
    BenchmarkPlotter.plot_benchmark(benchmark, output_path, **kwargs)
