import gym
import torch
import numpy as np
import sys
from gymnasium import spaces
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

# Add project root and challenge root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
challenge_root = project_root / "externals" / "macro-place-challenge-2026"
sys.path.insert(0, str(challenge_root))

from graphplace.models import Benchmark
from graphplace.graph.pyg_converter import to_hetero_data, parse_netlist_pb
from graphplace.legalize.legalize_challenge import push_apart, greedy_refine

from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from macro_place.utils import validate_placement

class PlacementEnv(gym.Env):
    """
    Gymnasium environment for Macro Placement.
    State: Observations contain macro positions and static features.
    Reward: Improvement in Proxy Cost after legalizing micro-overlaps.
    """
    def __init__(
        self,
        benchmark_name: str,
        max_steps: int = 50,
        testcase_root: str = "externals/MacroPlacement/Testcases/ICCAD04"
    ):
        super().__init__()
        self.benchmark_name = benchmark_name
        self.max_steps = max_steps
        
        # 1. Load Benchmark and PLC (Metadata) using challenge loader
        benchmark_dir = Path(testcase_root) / benchmark_name
        self.mp_benchmark, self.plc = load_benchmark_from_dir(benchmark_dir.as_posix())
        
        # 2. Extract connectivity check
        netlist_path = benchmark_dir / "netlist.pb.txt"
        from graphplace.graph.pyg_converter import parse_netlist_pb
        _, self.net_nodes, _ = parse_netlist_pb(str(netlist_path))
        
        # 3. Define Observation Space
        num_macros = self.mp_benchmark.num_macros
        self.observation_space = spaces.Box(low=0, high=1, shape=(num_macros, 6), dtype=np.float32)
        
        # 4. Action Space
        # Continuous relative movement [dx, dy] for each macro
        self.action_space = spaces.Box(low=-1, high=1, shape=(num_macros, 2), dtype=np.float32)
        
        self.current_pos = self.mp_benchmark.macro_positions.clone()
        self.current_step = 0
        self.last_score = float('inf')
        self.best_score = float('inf')

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # Reset to initial positions (from benchmark)
        self.current_pos = self.mp_benchmark.macro_positions.clone()
        
        if options and 'warm_start_pos' in options:
            self.current_pos = options['warm_start_pos'].clone().float()

        
        self.last_score = self._get_score(self.current_pos)
        self.best_score = self.last_score
        
        return self._get_obs() # Gym reset returns only obs

    def step(self, action, legalize: bool = False):
        self.current_step += 1
        
        # 1. Apply Action (Scale action to a movement range, e.g., 0.1% of canvas for refinement)
        move_range = 0.001 * max(self.mp_benchmark.canvas_width, self.mp_benchmark.canvas_height)
        delta = torch.tensor(action, dtype=torch.float32).view(-1, 2) * move_range
        
        # Only move non-fixed macros
        new_pos = self.current_pos + delta
        fixed_mask = self.mp_benchmark.macro_fixed.unsqueeze(-1)
        self.current_pos = torch.where(fixed_mask, self.current_pos, new_pos)
        
        # Clamp to canvas
        self.current_pos[:, 0].clamp_(0, self.mp_benchmark.canvas_width)
        self.current_pos[:, 1].clamp_(0, self.mp_benchmark.canvas_height)
        
        # 2. Legalize (Only if requested, as it is very slow)
        if legalize:
            legalized_pos = greedy_refine(
                self.current_pos.clone(),
                self.mp_benchmark
            )
        else:
            legalized_pos = self.current_pos
        
        # 3. Calculate Score and Reward
        current_score = self._get_score(legalized_pos)
        
        # Reward is improvement. Scale up for better gradient signal.
        reward = (self.last_score - current_score) * 100.0
        
        # Bonus for absolute improvement over best seen so far
        if current_score < self.best_score:
            reward += 1.0 * (self.best_score - current_score) * 1000.0
            self.best_score = current_score
            
        self.last_score = current_score
        self.current_pos = legalized_pos # Update current to legalized
        
        done = self.current_step >= self.max_steps
        
        return self._get_obs(), float(reward), done, {"proxy_score": current_score}

    def _get_obs(self):
        # Normalize positions for observation
        obs = torch.zeros((self.mp_benchmark.num_macros, 6))
        obs[:, 0] = self.current_pos[:, 0] / (self.mp_benchmark.canvas_width if self.mp_benchmark.canvas_width > 0 else 1.0)
        obs[:, 1] = self.current_pos[:, 1] / (self.mp_benchmark.canvas_height if self.mp_benchmark.canvas_height > 0 else 1.0)
        obs[:, 2:4] = self.mp_benchmark.macro_sizes / torch.tensor([
            self.mp_benchmark.canvas_width if self.mp_benchmark.canvas_width > 0 else 1.0, 
            self.mp_benchmark.canvas_height if self.mp_benchmark.canvas_height > 0 else 1.0
        ])
        obs[:, 4] = self.mp_benchmark.macro_fixed.float()
        # [x, y, w, h, fixed, dummy_soft]
        return obs.numpy()

    def _get_score(self, placement, fast: bool = True):
        # 1. Fast Overlap Check (Vectorized)
        from graphplace.legalize.legalize_challenge import compute_overlap_pairs_vec
        ii, jj, ox, oy = compute_overlap_pairs_vec(placement, self.mp_benchmark)
        overlap_count = len(ii)
        total_overlap_area = (ox * oy).sum().item()
        
        # 2. Wirelength (Relatively fast)
        # We still need to set the placement for the PLC to get wirelength
        from macro_place.objective import _set_placement
        _set_placement(self.plc, placement, self.mp_benchmark)
        wirelength = self.plc.get_cost()
        
        if fast:
            # Skip heavy density and congestion maps
            proxy_cost = wirelength
            density_cost = 0.0
            congestion_cost = 0.0
        else:
            density_cost = self.plc.get_density_cost()
            congestion_cost = self.plc.get_congestion_cost()
            proxy_cost = wirelength + 0.5 * density_cost + 0.5 * congestion_cost
        
        # 3. Penalties
        # Reduce area penalty slightly so it doesn't overwhelm the wirelength score
        overlap_penalty = overlap_count * 1.0 + total_overlap_area * 0.5
        return proxy_cost + overlap_penalty
