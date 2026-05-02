import torch
from dataclasses import dataclass
from graphplace.models import Benchmark

@dataclass
class NetlistMetadata:
    """
    Minified netlist metadata as requested.
    """
    h_routing_capacity: float
    v_routing_capacity: float
    num_nets: int
    num_macros: int
    num_clusters: int
    canvas_width: float
    canvas_height: float
    grid_rows: int
    grid_cols: int

    @classmethod
    def from_benchmark(cls, benchmark: Benchmark) -> "NetlistMetadata":
        return cls(
            h_routing_capacity=benchmark.hroutes_per_micron,
            v_routing_capacity=benchmark.vroutes_per_micron,
            num_nets=benchmark.num_nets,
            num_macros=benchmark.num_hard_macros,
            num_clusters=benchmark.num_soft_macros,
            canvas_width=benchmark.canvas_width,
            canvas_height=benchmark.canvas_height,
            grid_rows=benchmark.grid_rows,
            grid_cols=benchmark.grid_cols
        )

