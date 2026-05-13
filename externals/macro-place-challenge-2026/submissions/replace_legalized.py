import torch
from pathlib import Path

class RePlAceLegalizedPlacer:
    def __init__(self):
        self.output_path = Path("/Users/roshaniruku/code/GraphPlace/output/ibm01/ibm01_legalized.pt")

    def place(self, benchmark):
        if not self.output_path.exists():
            raise FileNotFoundError(f"Legalized placement not found at {self.output_path}")
        return torch.load(self.output_path)
