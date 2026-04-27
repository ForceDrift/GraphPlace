import sys
import torch
from dataclasses import dataclass, field
from typing import List
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "macro-place-challenge-2026" / "macro_place"))
from benchmark import Benchmark

# Load a benchmark
benchmark_path = 'macro-place-challenge-2026/benchmarks/processed/public/ariane133_ng45.pt'
print(f"Loading benchmark from: {benchmark_path}")
benchmark = Benchmark.load(benchmark_path)

print(f"Design: {benchmark.name}")
print(f"Macros: {benchmark.num_macros}")
print(f"Nets: {benchmark.num_nets}")
print(f"Canvas: {benchmark.canvas_width:.2f} x {benchmark.canvas_height:.2f} mm")
