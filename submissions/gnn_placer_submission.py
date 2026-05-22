import os
import sys
from pathlib import Path
import torch

# Add project root to path so we can import graphplace
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from graphplace.gnn_placer import PlaceGNN
from graphplace.graph.pyg_converter import to_hetero_data, parse_netlist_pb
from graphplace.legalize.legalize_challenge import greedy_refine

class GNNPlacer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}

    def _get_model(self, bench_name):
        if bench_name not in self.models:
            model = PlaceGNN().to(self.device)
            # Use universal_last as requested
            model_path = project_root / "models" / "gnn_placer_universal_last.pth"
            
            if not model_path.exists():
                model_path = project_root / "models" / "gnn_placer_universal_best.pth"
            
            if model_path.exists():
                print(f"Loading weights from {model_path.name}")
                model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
            self.models[bench_name] = model
        return self.models[bench_name]

    def place(self, benchmark):
        model = self._get_model(benchmark.name)
        
        # Load local Benchmark class for PyG graph conversion
        from graphplace.models import Benchmark as GPBenchmark
        
        # Cast the challenge benchmark to our GPBenchmark (they share fields)
        gp_bench = GPBenchmark(
            name=benchmark.name,
            canvas_width=float(benchmark.canvas_width),
            canvas_height=float(benchmark.canvas_height),
            num_macros=benchmark.num_macros,
            macro_positions=benchmark.macro_positions.clone(),
            macro_sizes=benchmark.macro_sizes.clone(),
            macro_fixed=benchmark.macro_fixed.clone(),
            macro_names=benchmark.macro_names,
            num_nets=benchmark.num_nets,
            net_nodes=benchmark.net_nodes,
            net_weights=benchmark.net_weights.clone(),
            grid_rows=benchmark.grid_rows,
            grid_cols=benchmark.grid_cols,
            port_positions=benchmark.port_positions.clone(),
            macro_pin_offsets=benchmark.macro_pin_offsets,
            net_pin_nodes=benchmark.net_pin_nodes,
            hroutes_per_micron=benchmark.hroutes_per_micron,
            vroutes_per_micron=benchmark.vroutes_per_micron,
            hard_macro_indices=benchmark.hard_macro_indices,
            soft_macro_indices=benchmark.soft_macro_indices,
            num_hard_macros=benchmark.num_hard_macros,
            num_soft_macros=benchmark.num_soft_macros
        )
        
        # Build graph data directly from the benchmark object
        graph_data = to_hetero_data(gp_bench).to(self.device)
        
        # Current positions
        current_pos = benchmark.macro_positions.clone().to(self.device)
        move_range = 0.001 * max(benchmark.canvas_width, benchmark.canvas_height)
        fixed_mask = benchmark.macro_fixed.unsqueeze(-1).to(self.device)
        
        for step in range(50):
            # Construct proximity edges (k=5)
            with torch.no_grad():
                dist = torch.cdist(current_pos, current_pos)
                dist.fill_diagonal_(float('inf'))
                k = min(5, current_pos.size(0) - 1)
                if k > 0:
                    topk = dist.topk(k, largest=False)
                    indices = topk.indices
                    src = torch.arange(current_pos.size(0), device=self.device).unsqueeze(1).expand(-1, k).reshape(-1)
                    dst = indices.reshape(-1)
                    graph_data['macro', 'near', 'macro'].edge_index = torch.stack([src, dst], dim=0)
                else:
                    graph_data['macro', 'near', 'macro'].edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)

                # Normalize positions for GNN input
                norm_pos = current_pos.clone()
                norm_pos[:, 0] /= (benchmark.canvas_width if benchmark.canvas_width > 0 else 1.0)
                norm_pos[:, 1] /= (benchmark.canvas_height if benchmark.canvas_height > 0 else 1.0)
                graph_data['macro'].x[:, :2] = norm_pos

                _, mu = model(graph_data)
                
            delta = mu.view(-1, 2) * move_range
            new_pos = current_pos + delta
            new_pos = torch.where(fixed_mask, current_pos, new_pos)
            
            new_pos[:, 0].clamp_(0, benchmark.canvas_width)
            new_pos[:, 1].clamp_(0, benchmark.canvas_height)
            current_pos = new_pos
            
        # Final Legalization to guarantee 0 overlaps for the challenge evaluator
        legalized_pos = greedy_refine(current_pos.cpu(), gp_bench)
        return legalized_pos
