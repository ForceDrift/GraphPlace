import torch
import networkx as nx
import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt

def force_directed_placement(pt_path, out_path, iterations=50, k_factor=1.0):
    print(f"Loading {pt_path}...")
    data = torch.load(pt_path, weights_only=False)
    
    net_nodes = data['net_nodes']
    macro_names = data['macro_names']
    
    if len(net_nodes) == 0:
        # Fallback to protobuf
        from graphplace.dreamplace.run_dreamplace_from_pt import parse_pb_nets
        benchmark_name = Path(pt_path).stem
        pb_file = Path(f"externals/MacroPlacement/Testcases/ICCAD04/{benchmark_name}/netlist.pb.txt")
        print(f"    Parsing connectivity from {pb_file} ...")
        if pb_file.exists():
            net_nodes = parse_pb_nets(pb_file, macro_names)
            print(f"    Parsed {len(net_nodes)} nets connecting macros from protobuf")
            
    macro_sizes = data['macro_sizes']
    
    num_macros = data['num_macros']
    canvas_w = data['canvas_width']
    canvas_h = data['canvas_height']
    
    print(f"Loaded design: {num_macros} macros, {len(net_nodes)} nets.")
    
    # 1. Build Bipartite NetworkX graph
    B = nx.Graph()
    
    # Add macro nodes
    macro_nodes = [f"m{i}" for i in range(num_macros)]
    B.add_nodes_from(macro_nodes, bipartite=0)
    
    # Add net nodes
    net_node_names = [f"net{i}" for i in range(len(net_nodes))]
    B.add_nodes_from(net_node_names, bipartite=1)
    
    # Add edges between macros and nets
    for net_idx, net in enumerate(net_nodes):
        nodes = net.tolist() if hasattr(net, 'tolist') else list(net)
        net_name = f"net{net_idx}"
        for macro_idx in nodes:
            B.add_edge(f"m{macro_idx}", net_name)
                    
    print(f"Built bipartite graph with {B.number_of_nodes()} nodes ({num_macros} macros, {len(net_nodes)} nets) and {B.number_of_edges()} edges.")
    
    # 2. Run Fruchterman-Reingold on the bipartite graph
    print(f"Running force-directed placement (Fruchterman-Reingold) on bipartite graph (k={k_factor:.3f}, iters={iterations})...")
    # Using spectral layout as a seed for better global structure
    print("  Calculating spectral seed...")
    init_pos = nx.spectral_layout(B)
    
    print("  Running spring layout...")
    pos_dict = nx.spring_layout(B, k=k_factor, iterations=iterations, pos=init_pos)
    
    # Extract only macro positions
    macro_pos = torch.zeros((num_macros, 2))
    for i in range(num_macros):
        x, y = pos_dict[f"m{i}"]
        macro_pos[i, 0] = x
        macro_pos[i, 1] = y
        
    return B, pos_dict, macro_pos

def place_macros_from_bipartite(data, macro_pos_normalized):
    """
    Takes normalized macro positions (from bipartite layout) and places them 
    in a large area to avoid overlaps while maintaining spatiality.
    """
    from graphplace.core.legalizer import legalize
    
    num_macros = data['num_macros']
    macro_sizes = data['macro_sizes']
    macro_fixed = data['macro_fixed']
    
    # 1. Estimate a 'safe' canvas size to avoid congestion
    # We want a very large canvas so we don't 'worry' about width/length
    total_area = (macro_sizes[:, 0] * macro_sizes[:, 1]).sum().item()
    canvas_side = (total_area**0.5) * 5.0 # 5x the side length of the total area square
    
    # 2. Scale normalized [-1, 1] positions to the large canvas center
    # Map [-1, 1] to [canvas_side*0.1, canvas_side*0.9]
    center = canvas_side / 2
    scale = canvas_side * 0.4
    
    macro_positions = macro_pos_normalized * scale + center
    
    # 3. Run legalization to remove overlaps
    print(f"Legalizing {num_macros} macros on a {canvas_side:.0f}x{canvas_side:.0f} virtual canvas...")
    legal_pos = legalize(macro_positions, macro_sizes, macro_fixed, 
                         canvas_w=canvas_side, canvas_h=canvas_side,
                         max_iter=500, eps=1.0)
    
    return legal_pos, canvas_side

def visualize_placed_macros(data, legal_pos, canvas_size, out_png):
    print(f"Generating macro placement visualization {out_png}...")
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.set_facecolor('#0f0f0f')
    
    import matplotlib.patches as patches
    from matplotlib.collections import PatchCollection
    
    rects = []
    sizes = data['macro_sizes']
    
    for i in range(data['num_macros']):
        x_center = legal_pos[i, 0].item()
        y_center = legal_pos[i, 1].item()
        w = sizes[i, 0].item()
        h = sizes[i, 1].item()
        
        # Center to bottom-left
        bl_x = x_center - w/2
        bl_y = y_center - h/2
        
        rect = patches.Rectangle((bl_x, bl_y), w, h)
        rects.append(rect)
            
    ax.add_collection(PatchCollection(rects, facecolor='#3498db', alpha=0.8, edgecolor='white', linewidths=0.2))
    
    # Auto-scale view to fit the placed macros
    all_x = legal_pos[:, 0]
    all_y = legal_pos[:, 1]
    margin = max(sizes.max().item(), canvas_size * 0.05)
    ax.set_xlim(all_x.min().item() - margin, all_x.max().item() + margin)
    ax.set_ylim(all_y.min().item() - margin, all_y.max().item() + margin)
    
    ax.set_aspect('equal')
    plt.title("Macro Placement from Bipartite Layout (No Overlaps)", color='white', fontsize=20)
    plt.axis('off')
    
    print(f"Saving placement visualization to {out_png}...")
    plt.savefig(out_png, dpi=200, bbox_inches='tight', facecolor='#0f0f0f')
    print("Done!")

def fit_to_actual_canvas(data, current_pos):
    """
    Scales the current placement to fit within the actual design canvas 
    and legalizes it within those boundaries.
    """
    from graphplace.core.legalizer import legalize
    
    cw = data['canvas_width']
    ch = data['canvas_height']
    sizes = data['macro_sizes']
    fixed = data['macro_fixed']
    num_macros = data['num_macros']
    
    print(f"Fitting placement to actual canvas: {cw:.0f} x {ch:.0f}")
    
    # 1. Map current bounding box to canvas (preserving aspect ratio roughly)
    min_x, max_x = current_pos[:, 0].min(), current_pos[:, 0].max()
    min_y, max_y = current_pos[:, 1].min(), current_pos[:, 1].max()
    
    curr_w = max_x - min_x
    curr_h = max_y - min_y
    
    # Scale and center
    # We use non-uniform scaling here to ensure we utilize the full canvas 
    # and reduce congestion by stretching the clusters.
    margin = 0.02
    target_w = cw * (1 - 2*margin)
    target_h = ch * (1 - 2*margin)
    
    scale_x = target_w / curr_w if curr_w > 0 else 1.0
    scale_y = target_h / curr_h if curr_h > 0 else 1.0
    
    new_pos = current_pos.clone()
    new_pos[:, 0] = (current_pos[:, 0] - (min_x + curr_w/2)) * scale_x + cw/2
    new_pos[:, 1] = (current_pos[:, 1] - (min_y + curr_h/2)) * scale_y + ch/2
    
    # 2. Legalize with actual canvas constraints
    print(f"Running final legalization on actual canvas (spread mode)...")
    final_pos = legalize(new_pos, sizes, fixed, cw, ch, 
                         max_iter=1000, eps=0.1)
    
    return final_pos

def visualize_bipartite(B, pos, out_png):
    print(f"Generating bipartite visualization {out_png}...")
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.set_facecolor('#0f0f0f')
    
    # Identify macro vs net nodes
    macro_nodes = [n for n, d in B.nodes(data=True) if d['bipartite'] == 0]
    net_nodes = [n for n, d in B.nodes(data=True) if d['bipartite'] == 1]
    
    # Draw edges with low alpha
    nx.draw_networkx_edges(B, pos, ax=ax, edge_color='#444444', alpha=0.2, width=0.5)
    
    # Draw macro nodes (larger, blue)
    nx.draw_networkx_nodes(B, pos, nodelist=macro_nodes, ax=ax, 
                           node_color='#3498db', node_size=20, alpha=0.8, label='Macros')
    
    # Draw net nodes (smaller, purple)
    nx.draw_networkx_nodes(B, pos, nodelist=net_nodes, ax=ax, 
                           node_color='#9b59b6', node_size=5, alpha=0.5, label='Nets')
    
    plt.title("Bipartite Force-Directed Layout (Untangled)", color='white', fontsize=20)
    plt.axis('off')
    # plt.legend(scatterpoints=1)
    
    print(f"Saving bipartite visualization to {out_png}...")
    plt.savefig(out_png, dpi=200, bbox_inches='tight', facecolor='#0f0f0f')
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt", required=True)
    parser.add_argument("--out-pt", required=True)
    parser.add_argument("--out-png", required=True)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--k", type=float, default=None, help="Spring constant k. Default: 1/sqrt(N)")
    args = parser.parse_args()
    
    # 1. Generate bipartite layout
    k_val = args.k if args.k is not None else (1.0 / (3478**0.5)) # Rough default for ibm01 nodes
    B, pos_dict, macro_pos_norm = force_directed_placement(args.pt, args.out_pt, iterations=args.iters, k_factor=k_val)
    visualize_bipartite(B, pos_dict, args.out_png.replace(".png", "_bipartite.png"))
    
    # 2. Place macros and remove overlaps
    data = torch.load(args.pt, weights_only=False)
    legal_pos, canvas_size = place_macros_from_bipartite(data, macro_pos_norm)
    
    # 3. Visualize intermediate large placement
    # visualize_placed_macros(data, legal_pos, canvas_size, args.out_png.replace(".png", "_expanded.png"))
    
    # 4. Fit to actual canvas
    final_pos = fit_to_actual_canvas(data, legal_pos)
    
    # 5. Visualize final result
    visualize_placed_macros(data, final_pos, data['canvas_height'], args.out_png)
    
    # 6. Save result
    data['macro_positions'] = final_pos
    torch.save(data, args.out_pt)
    print(f"Final fitted placement saved to {args.out_pt}")
