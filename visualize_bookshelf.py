import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import sys
import os
import argparse

def parse_nodes(filename):
    print(f"Parsing {filename}...")
    nodes = {}
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("NumNodes") or line.startswith("NumTerminals") or line.startswith("UCLA") or line.startswith("#") or not line:
                continue
            parts = line.split()
            name = parts[0]
            w = float(parts[1])
            h = float(parts[2])
            is_terminal = "terminal" in line
            nodes[name] = {'w': w, 'h': h, 'terminal': is_terminal}
    return nodes

def parse_pl(filename):
    print(f"Parsing {filename}...")
    pl = {}
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("UCLA") or line.startswith("#") or not line:
                continue
            parts = line.split()
            if len(parts) < 3: continue
            name = parts[0]
            x = float(parts[1])
            y = float(parts[2])
            pl[name] = (x, y)
    return pl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=str, default="adaptec1")
    parser.add_argument("--nodes", type=str)
    parser.add_argument("--pl", type=str)
    parser.add_argument("--out", type=str)
    args = parser.parse_args()

    design = args.design
    nodes_file = args.nodes or f"{design}.nodes"
    pl_file = args.pl or f"{design}.gp.pl"
    out_file = args.out or f"{design}_bookshelf_vis.png"

    if not os.path.exists(nodes_file):
        print(f"Error: {nodes_file} not found.")
        return
    if not os.path.exists(pl_file):
        print(f"Error: {pl_file} not found.")
        return

    nodes = parse_nodes(nodes_file)
    pl = parse_pl(pl_file)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_facecolor('#1e1e1e')

    rects_std = []
    rects_macro = []
    rects_term = []

    print("Building patches...")
    for name, pos in pl.items():
        if name not in nodes: continue
        node = nodes[name]
        x, y = pos
        w, h = node['w'], node['h']
        
        rect = patches.Rectangle((x, y), w, h)
        
        # Heuristic: large nodes are macros
        if node['terminal']:
            rects_term.append(rect)
        elif w > 100 or h > 100:
            rects_macro.append(rect)
        else:
            rects_std.append(rect)

    print(f"Plotting {len(rects_std)} standard cells, {len(rects_macro)} macros, {len(rects_term)} terminals...")
    
    # Standard cells: small, subtle
    ax.add_collection(PatchCollection(rects_std, facecolor='#9b59b6', alpha=0.1, linewidths=0))
    
    # Macros: larger, distinct
    ax.add_collection(PatchCollection(rects_macro, facecolor='#3498db', alpha=0.8, edgecolor='white', linewidths=0.5))
    
    # Terminals: fixed, highlight
    ax.add_collection(PatchCollection(rects_term, facecolor='#e74c3c', alpha=1.0, edgecolor='white', linewidths=1))

    ax.autoscale_view()
    ax.set_aspect('equal')
    plt.title(f"DREAMPlace: adaptec1", color='white')
    plt.tight_layout()
    
    print(f"Saving to {out_file}...")
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    print("Done!")

if __name__ == "__main__":
    main()
