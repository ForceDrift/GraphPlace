import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
	sys.path.insert(0, str(project_root))

from tqdm import tqdm as _tqdm

from models import Benchmark
from objective import compute_proxy_cost
from _plc import PlacementCost
from graphplace.legalizer import Legalizer

def set_global_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)


def parse_int_list(value: str) -> List[int]:
	items = [v.strip() for v in value.split(",") if v.strip()]
	return [int(v) for v in items]


def parse_float_list(value: str) -> List[float]:
	items = [v.strip() for v in value.split(",") if v.strip()]
	return [float(v) for v in items]


class VanillaPolicy(nn.Module):
	def __init__(self, in_dim: int = 6, hidden_dim: int = 128, out_dim: int = 2):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(in_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
		)
		self.mean_head = nn.Linear(hidden_dim, out_dim)
		self.log_std = nn.Parameter(torch.full((out_dim,), -1.0))

	def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		if x.dim() == 2:
			x = x.unsqueeze(0)
		emb = self.net(x)
		mean = self.mean_head(emb)
		log_std = self.log_std.view(1, 1, -1).expand_as(mean)
		if mean.shape[0] == 1:
			return mean[0], log_std[0]
		return mean, log_std


def build_macro_features(benchmark: Benchmark, positions: torch.Tensor) -> torch.Tensor:
	canvas_w = float(benchmark.canvas_width)
	canvas_h = float(benchmark.canvas_height)

	pos_norm = torch.stack(
		[positions[:, 0] / canvas_w, positions[:, 1] / canvas_h], dim=-1
	)
	size_norm = torch.stack(
		[
			benchmark.macro_sizes[:, 0] / canvas_w,
			benchmark.macro_sizes[:, 1] / canvas_h,
		],
		dim=-1,
	)
	fixed_flag = benchmark.macro_fixed.float().unsqueeze(-1)
	hard_flag = benchmark.get_hard_macro_mask().float().unsqueeze(-1)
	return torch.cat([pos_norm, size_norm, fixed_flag, hard_flag], dim=-1)


def clamp_positions(benchmark: Benchmark, positions: torch.Tensor) -> torch.Tensor:
	w2 = benchmark.macro_sizes[:, 0] / 2.0
	h2 = benchmark.macro_sizes[:, 1] / 2.0
	x = torch.clamp(positions[:, 0], w2, benchmark.canvas_width - w2)
	y = torch.clamp(positions[:, 1], h2, benchmark.canvas_height - h2)
	return torch.stack([x, y], dim=-1)


def apply_policy_update(
	benchmark: Benchmark,
	positions: torch.Tensor,
	deltas: torch.Tensor,
	step_scale: float,
) -> torch.Tensor:
	movable_mask = benchmark.get_movable_mask()
	updated = positions.clone()
	updated[movable_mask] = updated[movable_mask] + deltas[movable_mask] * step_scale
	return clamp_positions(benchmark, updated)


def legalize_positions(benchmark: Benchmark, positions: torch.Tensor) -> torch.Tensor:
	benchmark.macro_positions = positions.clone()
	legalizer = Legalizer(benchmark)
	legalizer.legalize()
	return benchmark.macro_positions.clone()


def sample_action(
	policy: VanillaPolicy, features: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	mean, log_std = policy(features)
	std = torch.exp(log_std)
	dist = Normal(mean, std)
	raw_action = dist.sample()
	action = torch.tanh(raw_action)
	log_prob = dist.log_prob(raw_action)
	log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-6)
	log_prob = log_prob.sum()
	return raw_action, action, log_prob


def get_netlist_path(project_root: Path, name: str) -> Optional[Path]:
	externals_root = project_root / "externals" / "MacroPlacement"

	if name.startswith("ibm"):
		return externals_root / "Testcases" / "ICCAD04" / name / "netlist.pb.txt"

	block = None
	if "ariane133" in name:
		block = "ariane133"
	elif "ariane136" in name:
		block = "ariane136"
	elif "mempool_tile" in name:
		block = "mempool_tile"
	elif "nvdla" in name:
		block = "nvdla"
	elif "bp_quad" in name:
		block = "bp_quad"

	if block is None:
		return None

	if "asap7" in name:
		tech = "ASAP7"
	elif "ng45" in name:
		tech = "NanGate45"
	else:
		return None

	return (
		externals_root
		/ "Flows"
		/ tech
		/ block
		/ "netlist"
		/ "output_CT_Grouping"
		/ "netlist.pb.txt"
	)


def init_plc(benchmark: Benchmark, netlist_path: Path) -> PlacementCost:
	plc = PlacementCost(str(netlist_path))
	plc.set_canvas_size(float(benchmark.canvas_width), float(benchmark.canvas_height))
	plc.set_placement_grid(int(benchmark.grid_cols), int(benchmark.grid_rows))
	plc.set_routes_per_micron(
		float(benchmark.hroutes_per_micron), float(benchmark.vroutes_per_micron)
	)
	# plc_client_os expects an integer smooth range for internal loops.
	plc.set_congestion_smooth_range(1)
	return plc


def collect_snapshots(
	benchmark: Benchmark,
	plc: PlacementCost,
	policy: VanillaPolicy,
	train_steps: int,
	snapshot_every: int,
	step_scale: float,
	lr: float,
	weights: Dict[str, float],
	baseline_momentum: float,
	show_progress: bool = True,
) -> Dict[str, torch.Tensor]:
	positions = benchmark.macro_positions.clone().float()
	positions = legalize_positions(benchmark, positions)
	policy.train()

	snapshots: List[torch.Tensor] = []
	proxy_costs: List[float] = []
	wirelength_costs: List[float] = []
	density_costs: List[float] = []
	congestion_costs: List[float] = []
	overlap_counts: List[int] = []
	total_overlap_areas: List[float] = []
	max_overlap_areas: List[float] = []
	num_macros_with_overlaps: List[int] = []
	overlap_ratios: List[float] = []
	snapshot_steps: List[int] = []

	optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
	global_step = 0
	baseline = 0.0

	def record(step: int, placement: torch.Tensor, metrics: Dict[str, float]) -> None:
		snapshots.append(placement.detach().cpu())
		proxy_costs.append(float(metrics["proxy_cost"]))
		wirelength_costs.append(float(metrics["wirelength_cost"]))
		density_costs.append(float(metrics["density_cost"]))
		congestion_costs.append(float(metrics["congestion_cost"]))
		overlap_counts.append(int(metrics["overlap_count"]))
		total_overlap_areas.append(float(metrics["total_overlap_area"]))
		max_overlap_areas.append(float(metrics["max_overlap_area"]))
		num_macros_with_overlaps.append(int(metrics["num_macros_with_overlaps"]))
		overlap_ratios.append(float(metrics["overlap_ratio"]))
		snapshot_steps.append(step)

	initial_metrics = compute_proxy_cost(positions, benchmark, plc, weights)
	print(
		f"{benchmark.name} step {global_step}: proxy_cost={initial_metrics['proxy_cost']:.4f}"
	)
	record(global_step, positions, initial_metrics)

	step_iter = range(1, train_steps + 1)
	if show_progress:
		step_iter = _tqdm(
			step_iter,
			desc=f"{benchmark.name} train",
			unit="step",
			leave=False,
		)

	for _ in step_iter:
		global_step += 1
		features = build_macro_features(benchmark, positions.detach())
		raw_action, action, log_prob = sample_action(policy, features)
		with torch.no_grad():
			positions = apply_policy_update(benchmark, positions, action, step_scale)
			positions = legalize_positions(benchmark, positions)
			metrics = compute_proxy_cost(positions, benchmark, plc, weights)
		reward = -float(metrics["proxy_cost"])
		baseline = (
			baseline_momentum * baseline + (1.0 - baseline_momentum) * reward
		)
		advantage = reward - baseline

		loss = -(advantage * log_prob)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		positions = positions.detach()

		print(
			f"{benchmark.name} step {global_step}: proxy_cost={metrics['proxy_cost']:.4f}"
		)
		if global_step % snapshot_every == 0:
			record(global_step, positions, metrics)

	return {
		"placements": torch.stack(snapshots, dim=0),
		"proxy_cost": torch.tensor(proxy_costs, dtype=torch.float32),
		"wirelength_cost": torch.tensor(wirelength_costs, dtype=torch.float32),
		"density_cost": torch.tensor(density_costs, dtype=torch.float32),
		"congestion_cost": torch.tensor(congestion_costs, dtype=torch.float32),
		"overlap_count": torch.tensor(overlap_counts, dtype=torch.int64),
		"total_overlap_area": torch.tensor(total_overlap_areas, dtype=torch.float32),
		"max_overlap_area": torch.tensor(max_overlap_areas, dtype=torch.float32),
		"num_macros_with_overlaps": torch.tensor(
			num_macros_with_overlaps, dtype=torch.int64
		),
		"overlap_ratio": torch.tensor(overlap_ratios, dtype=torch.float32),
		"snapshot_steps": torch.tensor(snapshot_steps, dtype=torch.int64),
	}


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Generate placement snapshots and proxy scores for all benchmarks."
	)
	parser.add_argument(
		"--data-dir",
		type=str,
		default="data/processed/public",
		help="Directory containing benchmark .pt files.",
	)
	parser.add_argument(
		"--out",
		type=str,
		default="data/generated/placement_dataset.pt",
		help="Output dataset path.",
	)
	parser.add_argument(
		"--train-steps",
		type=int,
		default=1000,
		help="Training steps per policy run.",
	)
	parser.add_argument(
		"--snapshot-every", type=int, default=5, help="Snapshot frequency."
	)
	parser.add_argument("--seed", type=int, default=42, help="Random seed.")
	parser.add_argument(
		"--step-scale",
		type=float,
		default=5.0,
		help="Step scale in microns for policy deltas.",
	)
	parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
	parser.add_argument(
		"--seeds",
		type=str,
		default="42",
		help="Comma-separated random seeds per benchmark.",
	)
	parser.add_argument(
		"--congestion-weights",
		type=str,
		default="0,0.25,0.5,0.75,1.0",
		help="Comma-separated congestion weights to sweep.",
	)
	parser.add_argument(
		"--baseline-momentum",
		type=float,
		default=0.9,
		help="EMA momentum for REINFORCE baseline.",
	)
	parser.add_argument(
		"--bench",
		type=str,
		default=None,
		help="Optional benchmark name to run (e.g., ariane133_ng45).",
	)

	args = parser.parse_args()
	set_global_seed(args.seed)
	seed_list = parse_int_list(args.seeds)
	congestion_weights = parse_float_list(args.congestion_weights)

	data_dir = project_root / args.data_dir
	if not data_dir.exists():
		raise FileNotFoundError(f"Data directory not found: {data_dir}")

	benchmark_paths = sorted(data_dir.glob("*.pt"))
	if args.bench:
		benchmark_paths = [p for p in benchmark_paths if p.stem == args.bench]
	if not benchmark_paths:
		raise FileNotFoundError(
			f"No .pt benchmarks found in {data_dir} for bench={args.bench}"
		)

	dataset: Dict[str, Dict[str, torch.Tensor]] = {}
	skipped: List[Tuple[str, str]] = []

	output_path = project_root / args.out
	output_path.parent.mkdir(parents=True, exist_ok=True)

	bench_iter = _tqdm(benchmark_paths, desc="Benchmarks", unit="bench")
	for benchmark_path in bench_iter:
		benchmark = Benchmark.load(str(benchmark_path))
		name = benchmark.name


		netlist_path = get_netlist_path(project_root, name)
		if netlist_path is None or not netlist_path.exists():
			skipped.append((name, "netlist.pb.txt not found"))
			continue

		plc = init_plc(benchmark, netlist_path)

		for seed in seed_list:
			set_global_seed(seed)
			for congestion_weight in congestion_weights:
				policy = VanillaPolicy()
				weights = {
					"wirelength": 1.0,
					"density": 0.5,
					"congestion": congestion_weight,
				}
				run_name = (
					f"{name}_seed{seed}_cw{congestion_weight:.2f}"
				)
				snapshots = collect_snapshots(
					benchmark=benchmark,
					plc=plc,
					policy=policy,
					train_steps=args.train_steps,
					snapshot_every=args.snapshot_every,
					step_scale=args.step_scale,
					lr=args.lr,
					weights=weights,
					baseline_momentum=args.baseline_momentum,
					show_progress=True,
				)

				dataset[run_name] = {
					"benchmark_path": str(benchmark_path),
					"netlist_path": str(netlist_path),
					"canvas_width": torch.tensor(benchmark.canvas_width),
					"canvas_height": torch.tensor(benchmark.canvas_height),
					"macro_sizes": benchmark.macro_sizes.clone().cpu(),
					"macro_fixed": benchmark.macro_fixed.clone().cpu(),
					"macro_names": benchmark.macro_names,
					"num_hard_macros": torch.tensor(benchmark.num_hard_macros),
					"num_soft_macros": torch.tensor(benchmark.num_soft_macros),
					"seed": seed,
					"congestion_weight": congestion_weight,
					**snapshots,
				}

				print(
					f"{run_name}: {snapshots['placements'].shape[0]} snapshots saved "
					f"from {benchmark_path.name}"
				)
				proxy_min = snapshots["proxy_cost"].min().item()
				proxy_max = snapshots["proxy_cost"].max().item()
				print(
					f"{run_name}: proxy_cost range = [{proxy_min:.4f}, {proxy_max:.4f}]"
				)

		torch.save(
			{
				"seed": args.seed,
				"train_steps": args.train_steps,
				"snapshot_every": args.snapshot_every,
				"step_scale": args.step_scale,
				"lr": args.lr,
				"seeds": seed_list,
				"congestion_weights": congestion_weights,
				"baseline_momentum": args.baseline_momentum,
				"benchmarks": dataset,
				"skipped": skipped,
			},
			output_path,
		)

	if dataset or skipped:
		print(f"Dataset checkpoint saved to {output_path}")

	if skipped:
		print("Skipped benchmarks:")
		for name, reason in skipped:
			print(f"  - {name}: {reason}")

	print(f"Dataset saved to {output_path}")


if __name__ == "__main__":
	main()
