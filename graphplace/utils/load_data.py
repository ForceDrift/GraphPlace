import sys
import torch
from pathlib import Path

# Add the project root to sys.path to allow absolute imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from graphplace.core.models import Benchmark
# from graphplace.visualization.plotter import BenchmarkPlotter # Fixed path
try:
    from graphplace.utils.vis import BenchmarkPlotter
except ImportError:
    BenchmarkPlotter = None

def load_benchmark(name: str = "ibm01"):
    """
    Loads a benchmark by name and returns the Benchmark object.
    """
    base_path = project_root / "data" / "processed" / "public"
    benchmark_path = base_path / f"{name}.pt"
    
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_path}")
        
    print(f"Loading benchmark from: {benchmark_path}")
    return Benchmark.load(str(benchmark_path))

if __name__ == "__main__":
    benchmark = load_benchmark("ibm01")
    
    print("\n--- Benchmark Summary ---")
    print(benchmark)
    
    print(f"\nNum Nets: {benchmark.num_nets}")
    if benchmark.num_nets > 0:
        print(f"First Net Nodes: {benchmark.net_nodes[0]}")
    
    # Accessing X and Y coordinates via the new properties
    print(f"\nMacro Coordinates (First 5):")
    for i in range(min(5, benchmark.num_macros)):
        print(f"  Macro {i}: x={benchmark.x[i]:.2f}, y={benchmark.y[i]:.2f}")
        
    # 3. Visualize
    if BenchmarkPlotter:
        print("\nGenerating visualization...")
        BenchmarkPlotter.plot_benchmark(benchmark, "benchmark_plot.png")
