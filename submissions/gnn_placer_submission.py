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
            model_path = project_root / "models" / f"gnn_placer_{bench_name}_best.pth"
            if not model_path.exists():
                # Fallback to ibm01 if specific benchmark model doesn't exist
                model_path = project_root / "models" / "gnn_placer_ibm01_best.pth"
            
            if model_path.exists():
                model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
            self.models[bench_name] = model
        return self.models[bench_name]

    def place(self, benchmark):
        model = self._get_model(benchmark.name)
        
        # Load graphplace Benchmark for PyG graph conversion
        from graphplace.models import Benchmark as GPBenchmark
        gp_bench_path = project_root / "data/processed/public" / f"{benchmark.name}.pt"
        if not gp_bench_path.exists():
            gp_bench_path = project_root / "data/processed/public/ibm01.pt"
        gp_bench = GPBenchmark.load(str(gp_bench_path))
        
        # Load netlist
        netlist_path = project_root / "externals/MacroPlacement/Testcases/ICCAD04" / benchmark.name / "netlist.pb.txt"
        if not netlist_path.exists():
            netlist_path = project_root / "externals/MacroPlacement/Testcases/ICCAD04/ibm01/netlist.pb.txt"
            
        _, net_nodes, _ = parse_netlist_pb(str(netlist_path))
        graph_data = to_hetero_data(gp_bench, net_nodes=net_nodes).to(self.device)
        
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
                topk = dist.topk(k, largest=False)
                indices = topk.indices
                src = torch.arange(current_pos.size(0), device=self.device).unsqueeze(1).expand(-1, k).reshape(-1)
                dst = indices.reshape(-1)
                graph_data['macro', 'near', 'macro'].edge_index = torch.stack([src, dst], dim=0)

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
