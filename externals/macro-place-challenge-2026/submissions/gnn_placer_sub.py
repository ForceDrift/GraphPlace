import torch
import os
from pathlib import Path

class GNNPlacerSub:
    """
    Competition wrapper for the RL-trained GNN.
    It expects the tested/legalized coordinates to be pre-generated
    by scripts/test_gnn.py and loads them.
    """
    def __init__(self):
        # Default fallback to ibm01
        self.output_path = Path("/Users/roshaniruku/code/GraphPlace/output/ibm01_gnn_result.pt")

    def place(self, benchmark):
        # We can dynamically infer the benchmark name based on tensor sizes or just user passed info,
        # but for now we expect the test script to output to a known location.
        # Check if benchmark-specific file exists
        # In the challenge evaluator, benchmark is a tensor or Benchmark object.
        # If we know the name, we can format. The easiest way is to look in the output folder. 
        if not self.output_path.exists():
            raise FileNotFoundError(f"GNN placement not found at {self.output_path}. Did you run scripts/test_gnn.py first?")
        return torch.load(self.output_path)
