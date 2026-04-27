import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ..parser.models import Design

class DesignPlotter:
    """
    Visualization utilities for chip designs.
    """
    
    @staticmethod
    def plot_design(design: Design, output_path: str = "placement.png", max_nodes: int = 5000):
        """
        Plots the placement of nodes within the die area.
        
        Args:
            design: The Design object.
            output_path: Where to save the image.
            max_nodes: Limit the number of nodes plotted for performance (uses a random sample).
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Draw Die Boundary
        die_w = design.die_hx - design.die_lx
        die_h = design.die_hy - design.die_ly
        boundary = patches.Rectangle(
            (design.die_lx, design.die_ly), die_w, die_h,
            linewidth=2, edgecolor='black', facecolor='none', label='Die Boundary'
        )
        ax.add_patch(boundary)
        
        # Sample nodes if there are too many
        node_list = list(design.nodes.values())
        if len(node_list) > max_nodes:
            import random
            node_list = random.sample(node_list, max_nodes)
            print(f"Warning: Too many nodes ({len(design.nodes)}). Plotting a random sample of {max_nodes}.")

        # Draw Nodes
        for node in node_list:
            color = 'red' if node.is_fixed else 'blue'
            alpha = 0.6 if not node.is_fixed else 1.0
            
            rect = patches.Rectangle(
                (node.x, node.y), node.width, node.height,
                linewidth=0.5, edgecolor='none', facecolor=color, alpha=alpha
            )
            ax.add_patch(rect)
            
        ax.set_xlim(design.die_lx - die_w*0.05, design.die_hx + die_w*0.05)
        ax.set_ylim(design.die_ly - die_h*0.05, design.die_hy + die_h*0.05)
        ax.set_aspect('equal')
        ax.set_title(f"Placement Visualization: {design.name}\n(Blue: Movable, Red: Fixed)")
        ax.set_xlabel("X (microns)")
        ax.set_ylabel("Y (microns)")
        
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.savefig(output_path, dpi=150)
        print(f"Visualization saved to {output_path}")
        plt.close()
