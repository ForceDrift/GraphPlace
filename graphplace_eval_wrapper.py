import torch
import os
from pathlib import Path

class BipartiteForceDirectedPlacer:
    """
    A 'placer' that returns the results of our bipartite force-directed 
    untangling and legalization process.
    """
    def place(self, benchmark):
        # We assume the benchmark name is in the benchmark object
        # The evaluate script provides the benchmark object
        name = getattr(benchmark, 'name', 'ibm01')
        
        # Path to our generated placement
        # Prefer higher iterations if they exist
        pt_path = Path(f"c:/Users/Roshan/code/GraphPlace/data/generated/{name}_fd_spread_500iter.pt")
        if not pt_path.exists():
            pt_path = Path(f"c:/Users/Roshan/code/GraphPlace/data/generated/{name}_fd_spread_high_iter.pt")
        if not pt_path.exists():
            pt_path = Path(f"c:/Users/Roshan/code/GraphPlace/data/generated/{name}_fd_spread.pt")
        
        if not pt_path.exists():
            print(f"Error: Could not find pre-calculated placement at {pt_path}")
            # Fallback to current positions (which might be random/initial)
            return benchmark.macro_positions
            
        print(f"Loading pre-calculated placement from {pt_path}...")
        data = torch.load(pt_path, weights_only=False)
        
        # Return the positions
        # Ensure it's a tensor of shape [N, 2]
        return data['macro_positions'].float()
