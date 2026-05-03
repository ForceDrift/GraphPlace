from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from graphplace.core.models import Benchmark  # noqa: E402
from graphplace.core.objective import compute_proxy_cost  # noqa: E402
from graphplace.core._plc import PlacementCost  # noqa: E402
from graphplace.core.legalizer import Legalizer  # noqa: E402


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_int_list(value: str) -> List[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99          # discount factor
    gae_lambda: float = 0.95     # GAE λ
    clip_eps: float = 0.2        # PPO clipping ε
    value_coef: float = 0.5      # weight of value loss
    entropy_coef: float = 0.01   # weight of entropy bonus
    ppo_epochs: int = 4          # gradient epochs per rollout
    rollout_steps: int = 32      # steps collected before each update
    mini_batch_size: int = 16    # mini-batch size inside each epoch
    max_grad_norm: float = 0.5   # gradient clipping

class ActorCritic(nn.Module):

    def __init__(
        self,
        in_dim: int = 7,
        hidden_dim: int = 128,
        out_dim: int = 2,
    ) -> None:
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.actor_mean = nn.Linear(hidden_dim, out_dim)
        self.log_std = nn.Parameter(torch.full((out_dim,), -1.0))
        self.critic = nn.Linear(hidden_dim, 1)

        # Orthogonal initialisation 
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.trunk(x)                          # (N, hidden)
        mean = self.actor_mean(emb)                  # (N, 2)
        log_std = self.log_std.unsqueeze(0).expand_as(mean)  # (N, 2)
        value = self.critic(emb).mean()              # scalar
        return mean, log_std, value


def build_macro_features(
    benchmark: Benchmark,
    positions: torch.Tensor,
    degree_norm: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)

    pos_norm = torch.stack(
        [positions[:, 0] / cw, positions[:, 1] / ch], dim=-1
    )
    size_norm = torch.stack(
        [benchmark.macro_sizes[:, 0] / cw, benchmark.macro_sizes[:, 1] / ch],
        dim=-1,
    )
    fixed_flag = benchmark.macro_fixed.float().unsqueeze(-1)
    hard_flag = benchmark.get_hard_macro_mask().float().unsqueeze(-1)

    n = positions.shape[0]
    if degree_norm is None:
        degree_norm = torch.zeros(n, 1, dtype=torch.float32)
    else:
        degree_norm = degree_norm.float().unsqueeze(-1)

    return torch.cat([pos_norm, size_norm, fixed_flag, hard_flag, degree_norm], dim=-1)


def compute_degree_norm(benchmark: Benchmark) -> torch.Tensor:
    n = benchmark.macro_sizes.shape[0]
    try:
        degrees = torch.zeros(n, dtype=torch.float32)
        for net in benchmark.nets:
            for idx in net.macro_indices:
                degrees[idx] += 1.0
        max_deg = degrees.max().clamp(min=1.0)
        return degrees / max_deg
    except AttributeError:
        return torch.zeros(n, dtype=torch.float32)


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
    movable = benchmark.get_movable_mask()
    updated = positions.clone()
    updated[movable] = updated[movable] + deltas[movable] * step_scale
    return clamp_positions(benchmark, updated)


def legalize_positions(
    benchmark: Benchmark, positions: torch.Tensor, use_legalizer: bool = True
) -> torch.Tensor:
    benchmark.macro_positions = positions.clone()
    if use_legalizer:
        Legalizer(benchmark).legalize()
        result = benchmark.macro_positions.clone()
    else:
        result = positions.clone()

    # Re-clamp: legalizer may push macro centres outside the canvas boundary.
    clamped = clamp_positions(benchmark, result)
    oob_mask = (clamped - result).abs().sum(dim=-1) > 1e-4
    if oob_mask.any():
        oob_indices = oob_mask.nonzero(as_tuple=True)[0].tolist()
        print(
            f"[legalize] Post-legalizer OOB on macro(s) {oob_indices[:8]}"
            f"{'…' if len(oob_indices)>8 else ''} — re-clamping."
        )
        result = clamped

    benchmark.macro_positions = result.clone()
    return result


@dataclass
class RolloutBuffer:
    log_probs: List[torch.Tensor] = field(default_factory=list)
    values: List[torch.Tensor] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    features: List[torch.Tensor] = field(default_factory=list)
    actions: List[torch.Tensor] = field(default_factory=list)   # raw (pre-tanh)

    def clear(self) -> None:
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.features.clear()
        self.actions.clear()

    def __len__(self) -> int:
        return len(self.rewards)


def compute_gae(
    rewards: List[float],
    values: List[torch.Tensor],
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    T = len(rewards)
    advantages = torch.zeros(T)
    gae = 0.0
    vals = [v.item() for v in values] + [last_value]

    for t in reversed(range(T)):
        delta = rewards[t] + gamma * vals[t + 1] - vals[t]
        gae = delta + gamma * gae_lambda * gae
        advantages[t] = gae

    returns = advantages + torch.tensor([v.item() for v in values])
    return advantages, returns


def ppo_update(
    policy: ActorCritic,
    optimizer: torch.optim.Optimizer,
    buffer: RolloutBuffer,
    last_value: float,
    cfg: PPOConfig,
) -> Dict[str, float]:
    advantages, returns = compute_gae(
        buffer.rewards, buffer.values, last_value,
        cfg.gamma, cfg.gae_lambda,
    )
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    T = len(buffer)
    old_log_probs = torch.stack(buffer.log_probs).detach()   # (T,)
    feat_list = buffer.features                              # list of (N,7)
    act_list = buffer.actions                                # list of (N,2)

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    n_updates = 0

    for _ in range(cfg.ppo_epochs):
        indices = torch.randperm(T)
        for start in range(0, T, cfg.mini_batch_size):
            mb_idx = indices[start: start + cfg.mini_batch_size]

            mb_policy_loss = torch.tensor(0.0)
            mb_value_loss = torch.tensor(0.0)
            mb_entropy = torch.tensor(0.0)

            for t in mb_idx.tolist():
                feat = feat_list[t]           # (N, 7)
                raw_act = act_list[t]         # (N, 2)

                mean, log_std, value = policy(feat)
                std = torch.exp(log_std)
                dist = Normal(mean, std)

                # log prob of the stored raw action
                lp = dist.log_prob(raw_act)
                lp = lp - torch.log(1 - torch.tanh(raw_act).pow(2) + 1e-6)
                new_log_prob = lp.sum()

                entropy = dist.entropy().sum()

                ratio = torch.exp(new_log_prob - old_log_probs[t])
                adv = advantages[t]

                # PPO clipped surrogate
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv
                policy_loss = -torch.min(surr1, surr2)

                value_loss = F.mse_loss(value, returns[t].detach())

                mb_policy_loss = mb_policy_loss + policy_loss
                mb_value_loss = mb_value_loss + value_loss
                mb_entropy = mb_entropy + entropy

            n = len(mb_idx)
            loss = (
                mb_policy_loss / n
                + cfg.value_coef * mb_value_loss / n
                - cfg.entropy_coef * mb_entropy / n
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()

            total_policy_loss += (mb_policy_loss / n).item()
            total_value_loss += (mb_value_loss / n).item()
            total_entropy += (mb_entropy / n).item()
            n_updates += 1

    denom = max(n_updates, 1)
    return {
        "policy_loss": total_policy_loss / denom,
        "value_loss": total_value_loss / denom,
        "entropy": total_entropy / denom,
    }


def collect_snapshots(
    benchmark: Benchmark,
    plc: PlacementCost,
    policy: ActorCritic,
    train_steps: int,
    snapshot_every: int,
    step_scale: float,
    weights: Dict[str, float],
    cfg: PPOConfig,
    show_progress: bool = True,
) -> Dict[str, torch.Tensor]:
    positions = benchmark.macro_positions.clone().float()
    positions = legalize_positions(benchmark, positions, use_legalizer=True)
    degree_norm = compute_degree_norm(benchmark)
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

    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    buffer = RolloutBuffer()
    global_step = 0

    def record(step: int, placement: torch.Tensor, m: Dict) -> None:
        snapshots.append(placement.detach().cpu())
        proxy_costs.append(float(m["proxy_cost"]))
        wirelength_costs.append(float(m["wirelength_cost"]))
        density_costs.append(float(m["density_cost"]))
        congestion_costs.append(float(m["congestion_cost"]))
        overlap_counts.append(int(m["overlap_count"]))
        total_overlap_areas.append(float(m["total_overlap_area"]))
        max_overlap_areas.append(float(m["max_overlap_area"]))
        num_macros_with_overlaps.append(int(m["num_macros_with_overlaps"]))
        overlap_ratios.append(float(m["overlap_ratio"]))
        snapshot_steps.append(step)

    initial_metrics = compute_proxy_cost(positions, benchmark, plc, weights)
    print(
        f"{benchmark.name} step {global_step}: "
        f"proxy_cost={initial_metrics['proxy_cost']:.4f}"
    )
    record(global_step, positions, initial_metrics)

    step_iter = range(1, train_steps + 1)

    _STALE_PATIENCE = 5
    _stale_count = 0
    _last_cost: Optional[float] = None

    for _ in step_iter:
        global_step += 1

        features = build_macro_features(benchmark, positions.detach(), degree_norm)

        with torch.no_grad():
            mean, log_std, value = policy(features)

        std = torch.exp(log_std)
        dist = Normal(mean, std)
        raw_action = dist.sample()
        action = torch.tanh(raw_action)
        log_prob = dist.log_prob(raw_action)
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum()

        with torch.no_grad():
            positions = apply_policy_update(benchmark, positions, action, step_scale)
            positions = legalize_positions(benchmark, positions, use_legalizer=False)
            metrics = compute_proxy_cost(positions, benchmark, plc, weights)

        reward = -float(metrics["proxy_cost"])

        current_cost = float(metrics["proxy_cost"])
        if _last_cost is not None and abs(current_cost - _last_cost) < 1e-6:
            _stale_count += 1
        else:
            _stale_count = 0
        _last_cost = current_cost

        if _stale_count >= _STALE_PATIENCE:
            print(
                f"[stale] proxy_cost frozen at {current_cost:.4f} for "
                f"{_stale_count} steps — applying random kick to movable macros."
            )
            movable = benchmark.get_movable_mask()
            kick = torch.zeros_like(positions)
            kick[movable] = torch.randn(movable.sum(), 2) * step_scale * 3.0
            positions = clamp_positions(benchmark, positions + kick)
            positions = legalize_positions(benchmark, positions, use_legalizer=False)
            metrics = compute_proxy_cost(positions, benchmark, plc, weights)
            reward = -float(metrics["proxy_cost"])
            _stale_count = 0
            _last_cost = float(metrics["proxy_cost"])

        buffer.log_probs.append(log_prob)
        buffer.values.append(value)
        buffer.rewards.append(reward)
        buffer.features.append(features)
        buffer.actions.append(raw_action.detach())

        positions = positions.detach()

        if len(buffer) >= cfg.rollout_steps:
            with torch.no_grad():
                next_feat = build_macro_features(
                    benchmark, positions, degree_norm
                )
                _, _, last_val = policy(next_feat)
            ppo_losses = ppo_update(policy, optimizer, buffer, last_val.item(), cfg)
            buffer.clear()
            if show_progress:
                print(
                    f"{benchmark.name} step {global_step}: "
                    f"proxy_cost={metrics['proxy_cost']:.4f} "
                    f"pi={ppo_losses['policy_loss']:.3f} "
                    f"vf={ppo_losses['value_loss']:.3f}",
                    flush=True,
                )

        if show_progress:
            print(
                f"{benchmark.name} step {global_step}: "
                f"proxy_cost={metrics['proxy_cost']:.4f}",
                flush=True,
            )

        if global_step % snapshot_every == 0:
            record(global_step, positions, metrics)

    if len(buffer) > 0:
        with torch.no_grad():
            next_feat = build_macro_features(benchmark, positions, degree_norm)
            _, _, last_val = policy(next_feat)
        ppo_update(policy, optimizer, buffer, last_val.item(), cfg)
        buffer.clear()

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


def get_netlist_path(root: Path, name: str) -> Optional[Path]:
    ext = root / "externals" / "MacroPlacement"

    if name.startswith("ibm"):
        return ext / "Testcases" / "ICCAD04" / name / "netlist.pb.txt"

    block: Optional[str] = None
    for key in ("ariane133", "ariane136", "mempool_tile", "nvdla", "bp_quad"):
        if key in name:
            block = key
            break
    if block is None:
        return None

    if "asap7" in name:
        tech = "ASAP7"
    elif "ng45" in name:
        tech = "NanGate45"
    else:
        return None

    return (
        ext / "Flows" / tech / block / "netlist" / "output_CT_Grouping" / "netlist.pb.txt"
    )


def init_plc(benchmark: Benchmark, netlist_path: Path) -> PlacementCost:
    plc = PlacementCost(str(netlist_path))
    plc.set_canvas_size(float(benchmark.canvas_width), float(benchmark.canvas_height))
    plc.set_placement_grid(int(benchmark.grid_cols), int(benchmark.grid_rows))
    plc.set_routes_per_micron(
        float(benchmark.hroutes_per_micron), float(benchmark.vroutes_per_micron)
    )
    plc.set_congestion_smooth_range(1)
    return plc


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate placement snapshots using a PPO policy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", default="data/processed/public")
    p.add_argument("--out", default="data/generated/placement_dataset.pt")
    p.add_argument("--train-steps", type=int, default=1000)
    p.add_argument("--snapshot-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--step-scale", type=float, default=5.0)
    p.add_argument("--seeds", type=str, default="42",
                   help="Comma-separated random seeds per benchmark.")
    p.add_argument("--bench", type=str, default=None,
                   help="Run a single benchmark by name.")
    p.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Optional directory to save policy checkpoints.",
    )

    ppo = p.add_argument_group("PPO")
    ppo.add_argument("--lr", type=float, default=3e-4)
    ppo.add_argument("--gamma", type=float, default=0.99)
    ppo.add_argument("--gae-lambda", type=float, default=0.95)
    ppo.add_argument("--clip-eps", type=float, default=0.2)
    ppo.add_argument("--value-coef", type=float, default=0.5)
    ppo.add_argument("--entropy-coef", type=float, default=0.01)
    ppo.add_argument("--ppo-epochs", type=int, default=4)
    ppo.add_argument("--rollout-steps", type=int, default=32)
    ppo.add_argument("--mini-batch-size", type=int, default=16)
    ppo.add_argument("--max-grad-norm", type=float, default=0.5)

    return p


def save_policy_checkpoint(policy: ActorCritic, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": policy.state_dict()}, path)


def generate_dataset(
    data_dir: Path,
    out_path: Path,
    train_steps: int,
    snapshot_every: int,
    step_scale: float,
    seed_list: List[int],
    cfg: PPOConfig,
    bench: Optional[str] = None,
    checkpoint_dir: Optional[Path] = None,
    show_progress: bool = True,
) -> Tuple[Dict[str, Dict], List[Tuple[str, str]]]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    benchmark_paths = sorted(data_dir.glob("*.pt"))
    if bench:
        benchmark_paths = [p for p in benchmark_paths if p.stem == bench]
    if not benchmark_paths:
        raise FileNotFoundError(
            f"No .pt benchmarks found in {data_dir} for bench={bench}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    dataset: Dict[str, Dict] = {}
    skipped: List[Tuple[str, str]] = []

    for benchmark_path in benchmark_paths:
        benchmark = Benchmark.load(str(benchmark_path))
        name = benchmark.name

        netlist_path = get_netlist_path(project_root, name)
        if netlist_path is None or not netlist_path.exists():
            skipped.append((name, "netlist.pb.txt not found"))
            continue

        plc = init_plc(benchmark, netlist_path)

        for seed in seed_list:
            set_global_seed(seed)
            policy = ActorCritic()
            weights = {
                "wirelength": 1.0,
                "density": 0.5,
                "congestion": 0.5,
            }
            run_name = f"{name}_seed{seed}"

            snapshots = collect_snapshots(
                benchmark=benchmark,
                plc=plc,
                policy=policy,
                train_steps=train_steps,
                snapshot_every=snapshot_every,
                step_scale=step_scale,
                weights=weights,
                cfg=cfg,
                show_progress=show_progress,
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
                **snapshots,
            }

            if checkpoint_dir is not None:
                checkpoint_path = checkpoint_dir / f"{run_name}_policy.pt"
                save_policy_checkpoint(policy, checkpoint_path)

            proxy_min = snapshots["proxy_cost"].min().item()
            proxy_max = snapshots["proxy_cost"].max().item()
            if show_progress:
                print(
                    f"{run_name}: {snapshots['placements'].shape[0]} snapshots | "
                    f"proxy_cost ∈ [{proxy_min:.4f}, {proxy_max:.4f}]",
                    flush=True,
                )

        torch.save(
            {
                "train_steps": train_steps,
                "snapshot_every": snapshot_every,
                "step_scale": step_scale,
                "seeds": seed_list,
                "ppo_config": cfg.__dict__,
                "benchmarks": dataset,
                "skipped": skipped,
            },
            out_path,
        )

    if show_progress:
        print(f"\nDataset saved to {out_path}", flush=True)
        if skipped:
            print("Skipped:", flush=True)
            for sname, reason in skipped:
                print(f"  {sname}: {reason}", flush=True)

    return dataset, skipped


def main() -> None:
    args = build_parser().parse_args()
    set_global_seed(args.seed)

    cfg = PPOConfig(
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        ppo_epochs=args.ppo_epochs,
        rollout_steps=args.rollout_steps,
        mini_batch_size=args.mini_batch_size,
        max_grad_norm=args.max_grad_norm,
    )

    seed_list = parse_int_list(args.seeds)
    data_dir = project_root / args.data_dir
    out_path = project_root / args.out
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None

    generate_dataset(
        data_dir=data_dir,
        out_path=out_path,
        train_steps=args.train_steps,
        snapshot_every=args.snapshot_every,
        step_scale=args.step_scale,
        seed_list=seed_list,
        cfg=cfg,
        bench=args.bench,
        checkpoint_dir=checkpoint_dir,
        show_progress=True,
    )


if __name__ == "__main__":
    main()