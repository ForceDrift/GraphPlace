#!/usr/bin/env python3
"""
Legalize RePlAce GP output using the macro-place-challenge-2026 framework.

Strategy:
  1. Load benchmark (.pt) to get canvas, grid and sizes.
  2. Parse RePlAce .pl (bottom-left coords) → convert to center coords.
  3. Snap each macro center to the nearest integer grid cell.
  4. Iteratively resolve overlaps by shifting conflicting macros to nearest free slot.
  5. Evaluate proxy cost before and after.

Usage:
  python3 scripts/legalize_challenge.py \\
      --pl output/ibm01/ibm01_gp.pl \\
      --benchmark ibm01 \\
      --output output/ibm01/ibm01_legalized.pt
"""

import sys
import os
import argparse
import torch
import math
import random

BASE = '/Users/roshaniruku/code/GraphPlace'
CHALLENGE = os.path.join(BASE, 'externals/macro-place-challenge-2026')
sys.path.insert(0, CHALLENGE)

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_proxy_cost
from macro_place.loader import load_benchmark_from_dir
from macro_place.utils import validate_placement


def parse_pl_bottomleft(pl_file):
    """Parse bookshelf .pl file → {name: (bl_x, bl_y)}"""
    pos = {}
    with open(pl_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('UCLA'):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                pos[parts[0]] = (float(parts[1]), float(parts[2]))
            except ValueError:
                continue
    return pos


def snap_to_grid(placement, benchmark):
    """
    Snap macro centers to the nearest grid-aligned center.
    Grid cell size = canvas / (cols, rows).
    """
    cw = benchmark.canvas_width
    ch = benchmark.canvas_height
    gc = benchmark.grid_cols
    gr = benchmark.grid_rows
    cell_w = cw / gc
    cell_h = ch / gr

    snapped = placement.clone()
    n = placement.shape[0]
    sizes = benchmark.macro_sizes    # [N, 2]

    for i in range(n):
        if benchmark.macro_fixed[i]:
            continue
        cx, cy = placement[i].tolist()
        w, h = sizes[i].tolist()

        # Grid of valid centers: center must be >= w/2서 from left, etc.
        # Simply snap center to nearest grid cell center
        col = round(cx / cell_w - 0.5)
        row = round(cy / cell_h - 0.5)
        col = max(0, min(col, gc - 1))
        row = max(0, min(row, gr - 1))

        snapped[i, 0] = (col + 0.5) * cell_w
        snapped[i, 1] = (row + 0.5) * cell_h

    return snapped


def compute_overlap_pairs_vec(placement, benchmark, threshold=0.0):
    """Vectorized overlap detection. Returns (idx_i, idx_j, ox, oy) tensors."""
    sizes = benchmark.macro_sizes   # [N, 2]
    pos = placement                  # [N, 2]
    n = pos.shape[0]

    # Pairwise differences
    xi = pos[:, 0].unsqueeze(1)   # [N,1]
    yi = pos[:, 1].unsqueeze(1)
    xj = pos[:, 0].unsqueeze(0)   # [1,N]
    yj = pos[:, 1].unsqueeze(0)

    dx = (xi - xj).abs()           # [N,N]
    dy = (yi - yj).abs()

    wi = sizes[:, 0].unsqueeze(1)
    hi = sizes[:, 1].unsqueeze(1)
    wj = sizes[:, 0].unsqueeze(0)
    hj = sizes[:, 1].unsqueeze(0)

    min_sep_x = (wi + wj) / 2.0
    min_sep_y = (hi + hj) / 2.0

    ox = (min_sep_x - dx).clamp(min=0)
    oy = (min_sep_y - dy).clamp(min=0)

    overlap = (ox > threshold) & (oy > threshold)
    # Only upper triangle, exclude diagonal
    mask = torch.triu(overlap, diagonal=1)

    ii, jj = mask.nonzero(as_tuple=True)
    return ii, jj, ox[ii, jj], oy[ii, jj]


def push_apart(placement, benchmark, max_iters=5000, eps=0.005, threshold=0.0):
    """
    Enhanced vectorized push-apart for strict legalization:
    - Moves macros apart by (overlap + eps).
    - Uses adaptive step size.
    - Implements stochastic axis selection to break 2D deadlocks.
    """
    sizes = benchmark.macro_sizes
    cw = benchmark.canvas_width
    ch = benchmark.canvas_height
    pl = placement.clone().float()
    fixed = benchmark.macro_fixed
    
    best_pl = pl.clone()
    min_overlaps = float('inf')

    for iteration in range(max_iters):
        ii, jj, ox, oy = compute_overlap_pairs_vec(pl, benchmark, threshold=threshold)
        num_ov = len(ii)
        
        if num_ov < min_overlaps:
            min_overlaps = num_ov
            best_pl = pl.clone()

        if num_ov == 0:
            print(f"  SUCCESS: All {num_ov} overlaps resolved in {iteration} iterations!")
            return pl

        if iteration % 100 == 0:
            print(f"  iter {iteration}: {num_ov} overlapping pairs (best so far: {min_overlaps})")

        # Dynamic epsilon: increases if stuck, decreases if moving
        cur_eps = eps * (1.0 + (iteration // 500) * 0.5)

        delta = torch.zeros_like(pl)
        count = torch.zeros(pl.shape[0], dtype=torch.float32)

        dx = pl[ii, 0] - pl[jj, 0]
        dy = pl[ii, 1] - pl[jj, 1]

        # Push along axis of minimum overlap
        push_x = (ox < oy).float()
        
        # Random axis swap for very close ties to break symmetry
        tie = (ox - oy).abs() < 1e-6
        if tie.any():
            push_x[tie] = (torch.rand(tie.sum()) > 0.5).float()
        push_y = 1.0 - push_x

        sign_x = torch.sign(dx)
        sign_x[sign_x == 0] = (torch.rand((sign_x == 0).sum()) * 2 - 1).sign()
        sign_y = torch.sign(dy)
        sign_y[sign_y == 0] = (torch.rand((sign_y == 0).sum()) * 2 - 1).sign()

        # Displacement per cell = half the overlap plus a safety buffer
        px = push_x * (ox * 0.5 + cur_eps)
        py = push_y * (oy * 0.5 + cur_eps)

        for idx, sign in [(ii, 1.0), (jj, -1.0)]:
            delta[:, 0].scatter_add_(0, idx, sign * sign_x * px)
            delta[:, 1].scatter_add_(0, idx, sign * sign_y * py)
            count.scatter_add_(0, idx, torch.ones(len(idx)))

        count = count.clamp(min=1)
        delta /= count.unsqueeze(1)
        
        delta[fixed] = 0.0
        pl = pl + delta

        # Keep inside canvas
        half_w = sizes[:, 0] / 2.0
        half_h = sizes[:, 1] / 2.0
        pl[:, 0] = pl[:, 0].clamp(min=half_w, max=cw - half_w)
        pl[:, 1] = pl[:, 1].clamp(min=half_h, max=ch - half_h)

    print(f"  Reached max_iters={max_iters}, returning best placement found ({min_overlaps} overlaps)")
    return best_pl


def greedy_refine(placement, benchmark, max_dist=2.0):
    """
    Final pass: For each overlapping macro, try small grid-aligned shifts 
    to see if we can find a zero-overlap position nearby.
    """
    pl = placement.clone()
    sizes = benchmark.macro_sizes
    cw, ch = benchmark.canvas_width, benchmark.canvas_height
    fixed = benchmark.macro_fixed
    num_hard = benchmark.num_hard_macros
    
    # Grid steps
    cell_w = cw / benchmark.grid_cols
    cell_h = ch / benchmark.grid_rows

    def get_overlaps(idx, pos_x, pos_y):
        # Check overlaps for macro idx at (pos_x, pos_y) against all other hard macros
        w_i, h_i = sizes[idx].tolist()
        li, ui = pos_x - w_i/2, pos_x + w_i/2
        bi, ti = pos_y - h_i/2, pos_y + h_i/2
        
        count = 0
        for j in range(num_hard):
            if idx == j: continue
            w_j, h_j = sizes[j].tolist()
            xj, yj = pl[j].tolist()
            lj, uj = xj - w_j/2, xj + w_j/2
            bj, tj = yj - h_j/2, yj + h_j/2
            
            if not (li >= uj or ui <= lj or bi >= tj or ti <= bj):
                count += 1
        return count

    # Identify overlapping hard macros
    overlap_indices = []
    for i in range(num_hard):
        if get_overlaps(i, pl[i, 0], pl[i, 1]) > 0:
            overlap_indices.append(i)
    
    if not overlap_indices:
        return pl

    print(f"  Greedy refining {len(overlap_indices)} macros...")
    
    # Run multiple passes
    for pass_idx in range(3):
        improved = 0
        overlap_indices = []
        for i in range(num_hard):
            if get_overlaps(i, pl[i, 0], pl[i, 1]) > 0:
                overlap_indices.append(i)
        
        if not overlap_indices: break

        for idx in overlap_indices:
            if fixed[idx]: continue
            
            orig_x, orig_y = pl[idx].tolist()
            best_pos = (orig_x, orig_y)
            min_ov = get_overlaps(idx, orig_x, orig_y)
            
            # Larger Search window (15 grid steps ~ 7um radius)
            steps = 15
            found = False
            for r in range(1, steps + 1):
                # search in expanding rings
                for dx in range(-r, r + 1):
                    for dy in range(-r, r + 1):
                        if abs(dx) != r and abs(dy) != r: continue
                        
                        tx = orig_x + dx * cell_w
                        ty = orig_y + dy * cell_h
                        
                        if tx < sizes[idx, 0]/2 or tx > cw - sizes[idx, 0]/2 or ty < sizes[idx, 1]/2 or ty > ch - sizes[idx, 1]/2:
                            continue
                        
                        ov = get_overlaps(idx, tx, ty)
                        if ov < min_ov:
                            min_ov = ov
                            best_pos = (tx, ty)
                            if ov == 0:
                                found = True
                                break
                    if found: break
                if found: break
            
            if best_pos != (orig_x, orig_y):
                pl[idx, 0], pl[idx, 1] = best_pos
                improved += 1
        
        print(f"    Pass {pass_idx+1}: improved {improved} macros")
        if improved == 0: break

    return pl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pl', required=True, help='RePlAce GP .pl file (bottom-left coords)')
    ap.add_argument('--benchmark', default='ibm01')
    ap.add_argument('--output', required=True, help='Output legalized .pt tensor')
    ap.add_argument('--no-eval', action='store_true')
    args = ap.parse_args()

    # Load benchmark
    pt_file = os.path.join(CHALLENGE, f'benchmarks/processed/public/{args.benchmark}.pt')
    benchmark = Benchmark.load(pt_file)
    print(f"Loaded {args.benchmark}: {benchmark.num_macros} macros, "
          f"canvas {benchmark.canvas_width:.2f}×{benchmark.canvas_height:.2f}, "
          f"grid {benchmark.grid_cols}×{benchmark.grid_rows}")

    # Load PlacementCost for evaluation
    source_dir = os.path.join(CHALLENGE, f'external/MacroPlacement/Testcases/ICCAD04/{args.benchmark}')
    _, plc = load_benchmark_from_dir(source_dir)

    # Parse GP placement (bottom-left) → center coords
    raw_pos = parse_pl_bottomleft(args.pl)
    placement = benchmark.macro_positions.clone()
    matched = 0
    for i, name in enumerate(benchmark.macro_names):
        if name in raw_pos:
            bl_x, bl_y = raw_pos[name]
            w, h = benchmark.macro_sizes[i].tolist()
            placement[i, 0] = bl_x + w / 2.0
            placement[i, 1] = bl_y + h / 2.0
            matched += 1
    print(f"Matched {matched}/{benchmark.num_macros} macros from .pl file")

    # ── Evaluate GP placement ──
    if not args.no_eval:
        gp_pairs_ii, _, _, _ = compute_overlap_pairs_vec(placement, benchmark, threshold=0.004)
        gp_costs = compute_proxy_cost(placement, benchmark, plc)
        print(f"\nGP placement:  proxy={gp_costs['proxy_cost']:.4f}  "
              f"WL={gp_costs['wirelength_cost']:.4f}  "
              f"density={gp_costs['density_cost']:.4f}  "
              f"cong={gp_costs['congestion_cost']:.4f}  "
              f"overlaps={len(gp_pairs_ii)}")

    # ── Step 1: Grid-snap ──
    print("\nSnapping to grid ...")
    snapped = snap_to_grid(placement, benchmark)
    snap_ii, _, _, _ = compute_overlap_pairs_vec(snapped, benchmark, threshold=0.004)
    print(f"  After snap: {len(snap_ii)} overlapping pairs")

    # ── Step 2: Push-apart ──
    print("\nResolving overlaps (push-apart) ...")
    legalized = push_apart(snapped, benchmark, max_iters=5000)

    # ── Step 3: Greedy Refinement ──
    print("\nLocal greedy refinement ...")
    legalized = greedy_refine(legalized, benchmark)

    # ── Evaluate legalized placement ──
    if not args.no_eval:
        is_valid, violations = validate_placement(legalized, benchmark)
        leg_costs = compute_proxy_cost(legalized, benchmark, plc)
        print(f"\nLegalized:     proxy={leg_costs['proxy_cost']:.4f}  "
              f"WL={leg_costs['wirelength_cost']:.4f}  "
              f"density={leg_costs['density_cost']:.4f}  "
              f"cong={leg_costs['congestion_cost']:.4f}  "
              f"overlaps={leg_costs['overlap_count']}")
        print(f"Valid={is_valid}")

    # ── Save ──
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(legalized, args.output)
    print(f"\nSaved legalized placement tensor → {args.output}")


if __name__ == '__main__':
    main()
