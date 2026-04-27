import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
from typing import Optional
from ..parser.models import Design

from matplotlib.collections import PolyCollection
import numpy as np

class DesignPlotter:
    """
    Visualization utilities for chip designs.
    """
    
    @staticmethod
    def plot_design(design: Design, output_path: str = "placement.png", max_nodes: Optional[int] = None):
        """
        Plots the placement of nodes within the die area.
        """
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Draw Die Boundary
        die_w = design.die_hx - design.die_lx
        die_h = design.die_hy - design.die_ly
        boundary = patches.Rectangle(
            (design.die_lx, design.die_ly), die_w, die_h,
            linewidth=2, edgecolor='black', facecolor='none', zorder=10
        )
        ax.add_patch(boundary)
        
        node_list = list(design.nodes.values())
        if max_nodes and len(node_list) > max_nodes:
            import random
            node_list = random.sample(node_list, max_nodes)
            print(f"Sampling {max_nodes} nodes for visualization.")
        else:
            print(f"Plotting all {len(node_list)} nodes. This may take a moment...")

        # Use PolyCollection for high-performance plotting of many rectangles
        verts = []
        colors = []
        
        for node in node_list:
            # Coordinates of the 4 corners
            v = [
                (node.x, node.y),
                (node.x + node.width, node.y),
                (node.x + node.width, node.y + node.height),
                (node.x, node.y + node.height)
            ]
            verts.append(v)
            colors.append((1, 0, 0, 1) if node.is_fixed else (0, 0.4, 1, 0.5))

        coll = PolyCollection(verts, facecolors=colors, edgecolors='none')
        ax.add_collection(coll)
            
        ax.set_xlim(design.die_lx - die_w*0.02, design.die_hx + die_w*0.02)
        ax.set_ylim(design.die_ly - die_h*0.02, design.die_hy + die_h*0.02)
        ax.set_aspect('equal')
        ax.set_title(f"Full Placement: {design.name}\n({len(node_list)} nodes)")
        
        plt.savefig(output_path, dpi=300) # Higher DPI for full plot
        print(f"Full visualization saved to {output_path}")
        plt.close()

    @staticmethod
    def plot_graph(G: nx.Graph, output_path: str = "graph_plot.png", max_edges: int = 10000):
        """
        Plots a NetworkX graph using the 'x' and 'y' attributes of nodes.
        """
        import random
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Get positions from node attributes
        pos = {node: (data['x'], data['y']) for node, data in G.nodes(data=True) if 'x' in data and 'y' in data}
        
        if not pos:
            print("Error: No positional data (x, y) found in graph nodes.")
            return

        # Plot Edges
        edges = list(G.edges())
        if len(edges) > max_edges:
            edges = random.sample(edges, max_edges)
            print(f"Sampling {max_edges} edges for graph plot.")
            
        from matplotlib.collections import LineCollection
        lines = [[pos[u], pos[v]] for u, v in edges if u in pos and v in pos]
        lc = LineCollection(lines, colors='gray', linewidths=0.5, alpha=0.3)
        ax.add_collection(lc)
        
        # Plot Nodes
        node_x = [p[0] for p in pos.values()]
        node_y = [p[1] for p in pos.values()]
        ax.scatter(node_x, node_y, s=1, c='blue', alpha=0.5)
        
        ax.set_aspect('equal')
        ax.set_title(f"Graph Connectivity Visualization\n({G.number_of_nodes()} nodes, {len(edges)} edges shown)")
        plt.savefig(output_path, dpi=300)
        print(f"Graph visualization saved to {output_path}")
        plt.close()
