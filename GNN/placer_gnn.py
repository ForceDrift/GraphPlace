"""
GNN Placer — placement-as-prediction, trained end-to-end via a
differentiable proxy cost (HPWL + density + overlap penalty).

Architecture
------------
  PlacerGNN
  ├── NodeEncoder      : project raw node features → hidden dim
  ├── GATConv layers   : graph attention message-passing
  └── PositionHead     : per-macro (x, y) ∈ (0,1)  (sigmoid)

Training loop
-------------
  for each epoch:
    1. Run GNN  →  predicted normalised positions
    2. Scale to canvas coords
    3. Compute differentiable proxy score (HPWL + density + overlap)
    4. Back-prop & update weights
"""

import sys
import time
import math
import argparse
from pathlib import Path
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

# ---------------------------------------------------------------------------
# path bootstrap
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from graphplace.models import Benchmark
from graphplace.graph.pyg_converter import to_pyg_data, parse_netlist_pb


# ============================================================
# 0.  NETLIST LOADER  (fills benchmark.net_nodes from .pb.txt)
# ============================================================

def _netlist_path(bench_name: str) -> Optional[Path]:
    """
    Resolve the netlist.pb.txt path for a given benchmark name,
    mirroring the logic in scripts/visualize_graph_cli.py.
    """
    ext = _ROOT / "externals" / "MacroPlacement"
    if bench_name.startswith("ibm"):
        return ext / "Testcases" / "ICCAD04" / bench_name / "netlist.pb.txt"

    block = None
    for key in ("ariane133", "ariane136", "mempool_tile", "nvdla", "bp_quad"):
        if key in bench_name:
            block = key
            break
    if block is None:
        return None

    tech = ("ASAP7"     if "asap7" in bench_name else
            "NanGate45" if "ng45"  in bench_name else None)
    if tech is None:
        return None

    return (ext / "Flows" / tech / block / "netlist" /
            "output_CT_Grouping" / "netlist.pb.txt")


def load_net_nodes(benchmark: Benchmark, bench_name: str) -> None:
    """
    Populate benchmark.net_nodes from the external netlist.pb.txt when
    the .pt file was saved without connectivity (net_nodes == []).

    Filters to nets that connect at least 2 macros, matching the
    behaviour of visualize_graph_cli.py.
    """
    if len(benchmark.net_nodes) > 0:
        return  # already populated

    path = _netlist_path(bench_name)
    if path is None or not path.exists():
        print(f"  [WARN] No netlist.pb.txt found for '{bench_name}' — "
              f"HPWL and congestion will be 0.")
        return

    print(f"  Loading connectivity from {path} ...")
    all_node_names, all_net_nodes, _ = parse_netlist_pb(str(path))

    name_to_idx = {n: i for i, n in enumerate(benchmark.macro_names)}
    filtered: List[torch.Tensor] = []
    for net in all_net_nodes:
        macro_indices = [
            name_to_idx[all_node_names[ni]]
            for ni in net
            if all_node_names[ni] in name_to_idx
        ]
        if len(macro_indices) >= 2:
            filtered.append(torch.tensor(macro_indices, dtype=torch.long))

    benchmark.net_nodes = filtered
    # Keep net_weights consistent (uniform weight for every net)
    benchmark.net_weights = torch.ones(len(filtered))
    print(f"  → {len(filtered)} nets connecting ≥2 macros loaded.")

# ============================================================
# 1.  GNN MODEL
# ============================================================

class PlacerGNN(nn.Module):
    """
    Predicts normalised (x, y) ∈ (0, 1) for every movable macro.

    The graph uses *star expansion* (net-nodes + macro-nodes) so
    net-level context naturally flows through message passing.
    """

    def __init__(
        self,
        in_channels: int,
        hidden: int = 128,
        num_layers: int = 4,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )

        # Stacked GAT layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            in_dim  = hidden if i == 0 else hidden
            out_dim = hidden // heads
            self.convs.append(
                GATConv(in_dim, out_dim, heads=heads, dropout=dropout, add_self_loops=False)
            )
            self.norms.append(nn.LayerNorm(hidden))

        self.dropout = dropout

        # Position prediction head (outputs for macro nodes only)
        self.pos_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 2),   # → (x, y)
        )

    def forward(self, data) -> torch.Tensor:
        """
        Args:
            data: torch_geometric.data.Data with .x, .edge_index, .edge_attr

        Returns:
            pos_norm: [num_nodes, 2] normalised positions (sigmoid output)
        """
        x = self.input_proj(data.x)

        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = conv(x, data.edge_index)
            x = norm(x + residual)
            x = F.gelu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        pos_norm = torch.sigmoid(self.pos_head(x))  # (0, 1)
        return pos_norm


# ============================================================
# 2.  DIFFERENTIABLE PROXY SCORE
# ============================================================

# ============================================================
# 2.  VECTORISED NET INDEX  (built once, reused every epoch)
# ============================================================

def build_net_index(
    net_nodes: List[torch.Tensor],
    net_weights: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Flatten the ragged net_nodes list into COO-style tensors.

    Returns:
        node_idx  : [E]  macro index for each membership entry
        net_idx   : [E]  net  index for each membership entry
        edge_w    : [E]  net weight for each membership entry

    Only nets with >= 2 members are included.
    """
    node_list, net_list, w_list = [], [], []
    for ni, net in enumerate(net_nodes):
        if len(net) < 2:
            continue
        node_list.append(net)
        net_list.append(torch.full((len(net),), ni, dtype=torch.long))
        w_list.append(torch.full((len(net),), float(net_weights[ni])))

    if not node_list:
        empty = torch.zeros(0, dtype=torch.long, device=device)
        return empty, empty, torch.zeros(0, dtype=torch.float, device=device)

    node_idx = torch.cat(node_list).to(device)
    net_idx  = torch.cat(net_list).to(device)
    edge_w   = torch.cat(w_list).to(device)
    return node_idx, net_idx, edge_w


# ============================================================
# 3.  DIFFERENTIABLE PROXY SCORE  (fully vectorised)
# ============================================================

def hpwl_loss(
    positions: torch.Tensor,
    node_idx: torch.Tensor,
    net_idx: torch.Tensor,
    edge_w: torch.Tensor,
    num_nets: int,
) -> torch.Tensor:
    """
    Smooth HPWL via log-sum-exp, fully vectorised over all nets.

    For each net we compute:
        HPWL_net = w * (soft_max(x) - soft_min(x) + soft_max(y) - soft_min(y))

    log-sum-exp scatter: for each net,
        soft_max(vals) = (1/α) * log Σ_i exp(α * val_i)
    implemented via torch.zeros(...).scatter_reduce.
    """
    if node_idx.numel() == 0:
        return torch.tensor(0.0, device=positions.device)

    alpha = 10.0
    device = positions.device
    # [E, 2]
    pts = positions[node_idx]          # coords of every membership entry

    total = torch.tensor(0.0, device=device, dtype=positions.dtype)
    for dim in range(2):
        vals = pts[:, dim]             # [E]

        # --- soft-max per net via log-sum-exp scatter ---
        scaled = alpha * vals          # [E]
        # per-net max for numerical stability
        max_per_net = torch.zeros(num_nets, device=device).scatter_reduce(
            0, net_idx, scaled, reduce='amax', include_self=True
        )                              # [N_nets]
        shifted = scaled - max_per_net[net_idx]   # [E]
        exp_sum = torch.zeros(num_nets, device=device).scatter_add(
            0, net_idx, shifted.exp()
        )                              # [N_nets]
        smax = (1.0 / alpha) * (max_per_net + exp_sum.clamp(min=1e-12).log())  # [N_nets]

        # --- soft-min per net = -soft_max(-vals) ---
        neg_scaled = -alpha * vals
        max_per_net_neg = torch.zeros(num_nets, device=device).scatter_reduce(
            0, net_idx, neg_scaled, reduce='amax', include_self=True
        )
        shifted_neg = neg_scaled - max_per_net_neg[net_idx]
        exp_sum_neg = torch.zeros(num_nets, device=device).scatter_add(
            0, net_idx, shifted_neg.exp()
        )
        smin = -(1.0 / alpha) * (max_per_net_neg + exp_sum_neg.clamp(min=1e-12).log())

        # per-net span weighted by net weight (edge_w has one entry per membership)
        # pick weight of first occurrence of each net (all entries same weight)
        net_w = torch.zeros(num_nets, device=device).scatter_reduce(
            0, net_idx, edge_w, reduce='amax', include_self=True
        )
        total = total + (net_w * (smax - smin)).sum()

    return total


def density_loss(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    canvas_w: float,
    canvas_h: float,
    grid: int = 32,
) -> torch.Tensor:
    """
    Density penalty: vectorised Gaussian spreading over all macros at once.
    """
    device = positions.device

    # Normalise to [0, 1]
    pos_n   = positions / torch.tensor([canvas_w, canvas_h], device=device)
    sizes_n = sizes      / torch.tensor([canvas_w, canvas_h], device=device)
    area    = (sizes_n[:, 0] * sizes_n[:, 1] + 1e-8).unsqueeze(1).unsqueeze(2)  # [N,1,1]

    bin_size = 1.0 / grid
    sigma    = bin_size * 2.0

    cx = torch.linspace(bin_size / 2, 1 - bin_size / 2, grid, device=device)  # [G]
    cy = torch.linspace(bin_size / 2, 1 - bin_size / 2, grid, device=device)  # [G]
    gx, gy = torch.meshgrid(cx, cy, indexing='ij')                            # [G, G]

    px = pos_n[:, 0].unsqueeze(1).unsqueeze(2)   # [N, 1, 1]
    py = pos_n[:, 1].unsqueeze(1).unsqueeze(2)   # [N, 1, 1]

    # [N, G, G] — Gaussian contribution of each macro to each bin
    contrib = torch.exp(-((gx - px) ** 2 + (gy - py) ** 2) / (2 * sigma ** 2)) * area
    density = contrib.sum(dim=0)                 # [G, G]

    target   = 1.0 / (grid * grid)
    overflow = F.relu(density - target)
    return overflow.sum()


def congestion_loss(
    positions: torch.Tensor,
    node_idx: torch.Tensor,
    net_idx: torch.Tensor,
    edge_w: torch.Tensor,
    num_nets: int,
    canvas_w: float,
    canvas_h: float,
    grid_h: int = 32,
    grid_v: int = 32,
) -> torch.Tensor:
    """
    Vectorised routing congestion cost.
    All per-net bbox spans and Gaussian kernels computed in one shot.
    """
    device = positions.device
    if node_idx.numel() == 0:
        return torch.tensor(0.0, device=device)

    pts = positions[node_idx]    # [E, 2]

    # --- per-net bbox spans (vectorised scatter min/max) ---
    INF = 1e9
    x_max = torch.full((num_nets,), -INF, device=device).scatter_reduce(
        0, net_idx, pts[:, 0], reduce='amax', include_self=True)
    x_min = torch.full((num_nets,),  INF, device=device).scatter_reduce(
        0, net_idx, pts[:, 0], reduce='amin', include_self=True)
    y_max = torch.full((num_nets,), -INF, device=device).scatter_reduce(
        0, net_idx, pts[:, 1], reduce='amax', include_self=True)
    y_min = torch.full((num_nets,),  INF, device=device).scatter_reduce(
        0, net_idx, pts[:, 1], reduce='amin', include_self=True)

    span_h = (x_max - x_min).clamp(min=1e-6)   # [N_nets]
    span_v = (y_max - y_min).clamp(min=1e-6)   # [N_nets]
    ctr_x  = (x_min + x_max) / 2.0             # [N_nets]
    ctr_y  = (y_min + y_max) / 2.0             # [N_nets]
    net_w  = torch.zeros(num_nets, device=device).scatter_reduce(
        0, net_idx, edge_w, reduce='amax', include_self=True)

    # --- Grid centres ---
    bin_w, bin_h = canvas_w / grid_h, canvas_h / grid_v
    sigma_h, sigma_v = bin_w * 1.5, bin_h * 1.5

    cx = torch.linspace(bin_w / 2, canvas_w - bin_w / 2, grid_h, device=device)
    cy = torch.linspace(bin_h / 2, canvas_h - bin_h / 2, grid_v, device=device)
    gx, gy = torch.meshgrid(cx, cy, indexing='ij')   # [Gh, Gv]

    # Vectorised kernel for all nets: [N_nets, Gh, Gv]
    ctr_x3 = ctr_x.view(-1, 1, 1)
    ctr_y3 = ctr_y.view(-1, 1, 1)
    kernel = torch.exp(
        -((gx - ctr_x3) ** 2) / (2 * sigma_h ** 2)
        - ((gy - ctr_y3) ** 2) / (2 * sigma_v ** 2)
    )                                               # [N_nets, Gh, Gv]
    kernel = kernel / (kernel.sum(dim=(1, 2), keepdim=True) + 1e-8)

    w_h = (net_w * span_h).view(-1, 1, 1)   # [N_nets, 1, 1]
    w_v = (net_w * span_v).view(-1, 1, 1)

    h_demand = (w_h * kernel).sum(dim=0)    # [Gh, Gv]
    v_demand = (w_v * kernel).sum(dim=0)

    h_overflow = F.relu(h_demand - bin_w)
    v_overflow = F.relu(v_demand - bin_h)

    norm = (canvas_w * canvas_h) + 1e-8
    return (h_overflow.sum() + v_overflow.sum()) / norm


def proxy_score(
    positions: torch.Tensor,
    node_idx: torch.Tensor,
    net_idx: torch.Tensor,
    edge_w: torch.Tensor,
    num_nets: int,
    sizes: torch.Tensor,
    canvas_w: float,
    canvas_h: float,
    w_hpwl: float = 1.0,
    w_density: float = 0.5,
    w_congestion: float = 0.5,
) -> Tuple[torch.Tensor, dict]:
    """
    Composite differentiable proxy cost.

    Formula (matching ISPD/Google Brain convention):
        proxy = 1.0 × Wirelength + 0.5 × Density + 0.5 × Congestion
    """
    # --- Wirelength: smooth HPWL, normalised by canvas diagonal ---
    hpwl = hpwl_loss(positions, node_idx, net_idx, edge_w, num_nets)
    diag = math.sqrt(canvas_w ** 2 + canvas_h ** 2) + 1e-8
    hpwl_n = hpwl / diag

    # --- Density: Gaussian-spread cell overflow ---
    dens = density_loss(positions, sizes, canvas_w, canvas_h, grid=32)

    # --- Congestion: routing demand vs supply per grid cell ---
    cong = congestion_loss(
        positions, node_idx, net_idx, edge_w, num_nets,
        canvas_w, canvas_h, grid_h=32, grid_v=32
    )

    total = w_hpwl * hpwl_n + w_density * dens + w_congestion * cong

    info = dict(
        total=total.item(),
        hpwl=hpwl_n.item(),
        density=dens.item(),
        congestion=cong.item(),
    )
    return total, info


# ============================================================
# 3.  GRAPH BUILDER  (benchmark → PyG Data with correct feature dim)
# ============================================================

def build_graph(benchmark: Benchmark, device: torch.device):
    """Convert Benchmark → PyG Data, move to device."""
    data = to_pyg_data(benchmark, expansion='star', include_positions=True)

    # Degree feature (normalised)
    num_macros = benchmark.num_macros
    num_nodes  = data.x.size(0)
    degrees    = torch.zeros(num_nodes, 1)

    for net in benchmark.net_nodes:
        for idx in net:
            if idx < num_macros:
                degrees[idx] += 1

    max_deg = degrees.max().clamp(min=1.0)
    degrees = degrees / max_deg

    # BBox span per net-node
    bbox = torch.zeros(num_nodes, 2)
    pos  = benchmark.macro_positions
    cw   = benchmark.canvas_width  or 1.0
    ch   = benchmark.canvas_height or 1.0

    for ni, net in enumerate(benchmark.net_nodes):
        if len(net) < 1:
            continue
        pts = pos[net]
        span_x = (pts[:, 0].max() - pts[:, 0].min()) / cw
        span_y = (pts[:, 1].max() - pts[:, 1].min()) / ch
        bbox[num_macros + ni] = torch.tensor([span_x, span_y])

    data.x = torch.cat([data.x, degrees, bbox], dim=-1)
    return data.to(device)


# ============================================================
# 4.  TRAINING LOOP
# ============================================================

def train(
    benchmark: Benchmark,
    epochs: int = 500,
    lr: float = 3e-4,
    hidden: int = 128,
    num_layers: int = 4,
    heads: int = 4,
    w_hpwl: float = 1.0,
    w_density: float = 0.5,
    w_congestion: float = 0.5,
    log_every: int = 25,
    device_str: str = "cpu",
):
    device = torch.device(device_str)

    print(f"\n{'='*60}")
    print(f"  GraphPlace GNN Placer")
    print(f"{'='*60}")
    print(f"  Benchmark : {benchmark.name}")
    print(f"  Macros    : {benchmark.num_hard_macros} hard + {benchmark.num_soft_macros} soft")
    print(f"  Nets      : {benchmark.num_nets}")
    print(f"  Canvas    : {benchmark.canvas_width:.1f} × {benchmark.canvas_height:.1f} µm")
    print(f"  Device    : {device}")
    print(f"  Epochs    : {epochs}  |  LR: {lr}  |  Hidden: {hidden}")
    print(f"{'='*60}\n")

    # ----- build graph (static topology) -----
    data = build_graph(benchmark, device)
    in_ch = data.x.size(-1)
    num_macros = benchmark.num_macros

    # ----- pre-build net index (once) -----
    node_idx, net_idx, edge_w = build_net_index(
        benchmark.net_nodes, benchmark.net_weights, device
    )
    num_nets_idx = len(benchmark.net_nodes)  # full count for scatter targets
    macro_sizes = benchmark.macro_sizes.to(device)
    print(f"  Net index built: {node_idx.numel()} membership entries across "
          f"{(net_idx.max().item()+1) if node_idx.numel() else 0} nets.")

    # ----- model -----
    model = PlacerGNN(in_channels=in_ch, hidden=hidden, num_layers=num_layers, heads=heads).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)

    cw = benchmark.canvas_width
    ch = benchmark.canvas_height
    fixed_mask  = benchmark.macro_fixed.to(device)   # True → fixed
    fixed_pos   = benchmark.macro_positions.to(device)  # reference for fixed

    history = []

    best_loss   = float('inf')
    best_state  = None

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()

        # ── Forward pass ──────────────────────────────────────────
        pos_norm = model(data)              # [num_nodes, 2]  (0,1)
        macro_pos_norm = pos_norm[:num_macros]  # macro predictions

        # Scale to canvas
        positions = torch.stack([
            macro_pos_norm[:, 0] * cw,
            macro_pos_norm[:, 1] * ch,
        ], dim=1)                           # [num_macros, 2]

        # Pin fixed macros at their reference positions
        if fixed_mask.any():
            fixed_ref = fixed_pos[fixed_mask]
            positions = positions.clone()
            positions[fixed_mask] = fixed_ref

        # ── Proxy score: 1.0 × WL + 0.5 × Density + 0.5 × Congestion ──
        loss, info = proxy_score(
            positions, node_idx, net_idx, edge_w, num_nets_idx,
            macro_sizes, cw, ch,
            w_hpwl=w_hpwl,
            w_density=w_density,
            w_congestion=w_congestion,
        )

        # ── Back-prop ─────────────────────────────────────────────
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        sched.step()

        history.append(info)

        if loss.item() < best_loss:
            best_loss  = loss.item()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % log_every == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(
                f"  Epoch {epoch:5d}/{epochs}"
                f"  loss={info['total']:.4f}"
                f"  hpwl={info['hpwl']:.4f}"
                f"  dens={info['density']:.4f}"
                f"  cong={info['congestion']:.4f}"
                f"  lr={sched.get_last_lr()[0]:.2e}"
                f"  [{elapsed:.1f}s]"
            )

    print(f"\n  Best proxy cost : {best_loss:.4f}")
    print(f"  Total time      : {time.time()-t0:.1f}s")

    # Restore best weights & get final placement
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pos_norm   = model(data)[:num_macros]
        positions  = torch.stack([pos_norm[:, 0] * cw, pos_norm[:, 1] * ch], dim=1)
        if fixed_mask.any():
            positions[fixed_mask] = fixed_pos[fixed_mask]

    return model, positions.cpu(), history


# ============================================================
# 5.  CLI ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train a GNN to place macros by minimising a proxy cost."
    )
    parser.add_argument("--bench",      type=str,   default="ibm01",
                        help="Benchmark name (must exist in --data-dir).")
    parser.add_argument("--data-dir",   type=str,   default="data/processed/public",
                        help="Directory containing .pt benchmark files.")
    parser.add_argument("--epochs",     type=int,   default=300,
                        help="Number of training epochs.")
    parser.add_argument("--lr",         type=float, default=3e-4,
                        help="Learning rate.")
    parser.add_argument("--hidden",     type=int,   default=128,
                        help="GNN hidden dimension.")
    parser.add_argument("--layers",     type=int,   default=4,
                        help="Number of GAT layers.")
    parser.add_argument("--heads",      type=int,   default=4,
                        help="Number of attention heads.")
    parser.add_argument("--w-hpwl",       type=float, default=1.0,
                        help="Wirelength (HPWL) loss weight (default: 1.0).")
    parser.add_argument("--w-density",    type=float, default=0.5,
                        help="Density overflow loss weight (default: 0.5).")
    parser.add_argument("--w-congestion", type=float, default=0.5,
                        help="Routing congestion loss weight (default: 0.5).")
    parser.add_argument("--log-every", type=int,   default=25,
                        help="Print log every N epochs.")
    parser.add_argument("--device",    type=str,   default="cpu",
                        help="Torch device string (cpu / cuda / mps).")
    parser.add_argument("--save",      type=str,   default=None,
                        help="Path to save final model checkpoint (.pt).")
    parser.add_argument("--out-placement", type=str, default=None,
                        help="Save final macro positions as a .pt tensor.")
    args = parser.parse_args()

    bench_path = Path(args.data_dir) / f"{args.bench}.pt"
    if not bench_path.exists():
        print(f"[ERROR] Benchmark not found: {bench_path}")
        sys.exit(1)

    benchmark = Benchmark.load(str(bench_path))
    load_net_nodes(benchmark, args.bench)

    model, final_positions, history = train(
        benchmark,
        epochs       = args.epochs,
        lr           = args.lr,
        hidden       = args.hidden,
        num_layers   = args.layers,
        heads        = args.heads,
        w_hpwl       = args.w_hpwl,
        w_density    = args.w_density,
        w_congestion = args.w_congestion,
        log_every    = args.log_every,
        device_str   = args.device,
    )

    if args.save:
        torch.save(model.state_dict(), args.save)
        print(f"  Model saved → {args.save}")

    if args.out_placement:
        torch.save(final_positions, args.out_placement)
        print(f"  Placement saved → {args.out_placement}")

    # Quick stats
    print(f"\n  Final placement summary:")
    print(f"    X range: {final_positions[:,0].min():.2f} – {final_positions[:,0].max():.2f}")
    print(f"    Y range: {final_positions[:,1].min():.2f} – {final_positions[:,1].max():.2f}")


if __name__ == "__main__":
    main()
