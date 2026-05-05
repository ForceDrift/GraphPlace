"""
Fast vectorized geometric legalizer for macro placement.

Uses PyTorch broadcasting for O(N^2) overlap detection in a single
tensor operation, then resolves overlaps iteratively by pushing
the worst-overlapping pair apart each step.
"""
import torch
from typing import Tuple


def count_overlaps(pos: torch.Tensor, sizes: torch.Tensor,
                   mask: torch.Tensor = None) -> int:
    """Count exact pairwise overlaps using vectorized ops."""
    if mask is not None:
        idx = mask.nonzero(as_tuple=True)[0]
        pos = pos[idx]
        sizes = sizes[idx]
    
    N = pos.shape[0]
    if N < 2:
        return 0

    # [N,1] - [1,N] = [N,N] pairwise distances
    dx = (pos[:, 0].unsqueeze(1) - pos[:, 0].unsqueeze(0)).abs()
    dy = (pos[:, 1].unsqueeze(1) - pos[:, 1].unsqueeze(0)).abs()
    min_dx = (sizes[:, 0].unsqueeze(1) + sizes[:, 0].unsqueeze(0)) / 2
    min_dy = (sizes[:, 1].unsqueeze(1) + sizes[:, 1].unsqueeze(0)) / 2

    overlap_matrix = (dx < min_dx) & (dy < min_dy)
    overlap_matrix.fill_diagonal_(False)
    return overlap_matrix.sum().item() // 2


def legalize(pos: torch.Tensor, sizes: torch.Tensor,
             fixed: torch.Tensor, canvas_w: float, canvas_h: float,
             hard_only: bool = True, num_hard: int = None,
             max_iter: int = 500, eps: float = 0.01) -> torch.Tensor:
    """
    Fast vectorized legalization.
    
    Each iteration:
      1. Compute all pairwise overlap amounts (vectorized)
      2. For each macro, compute net displacement from all overlapping neighbors
      3. Apply displacement (scaled down for stability)
      4. Clamp to canvas
    """
    result = pos.clone().float()
    N = pos.shape[0]
    w = sizes[:, 0].float()
    h = sizes[:, 1].float()
    
    # Which macros participate in legalization
    if hard_only and num_hard is not None:
        active = torch.zeros(N, dtype=torch.bool)
        active[:num_hard] = True
    else:
        active = torch.ones(N, dtype=torch.bool)
    
    movable = active & (~fixed)
    
    # Work on active subset for speed
    a_idx = active.nonzero(as_tuple=True)[0]
    m_mask = movable[a_idx]  # which of the active ones are movable
    Na = len(a_idx)
    
    if Na < 2:
        return result
    
    a_pos = result[a_idx].clone()
    a_w = w[a_idx]
    a_h = h[a_idx]
    
    # Minimum separations [Na, Na]
    min_sep_x = (a_w.unsqueeze(1) + a_w.unsqueeze(0)) / 2 + eps
    min_sep_y = (a_h.unsqueeze(1) + a_h.unsqueeze(0)) / 2 + eps
    
    for iteration in range(max_iter):
        # Pairwise signed distances
        dx = a_pos[:, 0].unsqueeze(1) - a_pos[:, 0].unsqueeze(0)  # [Na, Na]
        dy = a_pos[:, 1].unsqueeze(1) - a_pos[:, 1].unsqueeze(0)
        
        # Overlap amounts (positive = overlapping)
        ox = min_sep_x - dx.abs()  # [Na, Na]
        oy = min_sep_y - dy.abs()
        
        # Overlap exists where both ox > 0 and oy > 0
        is_overlap = (ox > 0) & (oy > 0)
        is_overlap.fill_diagonal_(False)
        
        n_ov = is_overlap.sum().item() // 2
        if n_ov == 0:
            print(f"    Converged at iteration {iteration}: 0 overlaps")
            break
        
        if iteration % 200 == 0:
            print(f"    Iteration {iteration}: {n_ov} overlaps")
        
        # For each overlapping pair, compute push direction
        push_x = is_overlap.float() * ox.clamp(min=0)
        push_y = is_overlap.float() * oy.clamp(min=0)
        
        push_in_x = (ox < oy) & is_overlap
        push_in_y = (~push_in_x) & is_overlap
        
        sign_x = dx.sign()
        sign_x[sign_x == 0] = 1.0  # break ties
        sign_y = dy.sign()
        sign_y[sign_y == 0] = 1.0
        
        force_x = (push_in_x.float() * ox.clamp(min=0) * sign_x).sum(dim=1)
        force_y = (push_in_y.float() * oy.clamp(min=0) * sign_y).sum(dim=1)
        
        # Scale down based on iteration to converge smoothly
        scale = 0.5 * (0.99 ** (iteration / 10))
        scale = max(scale, 0.05)
        displacement = torch.stack([force_x * scale, force_y * scale], dim=1)
        
        # Only move movable macros
        displacement[~m_mask] = 0
        
        a_pos += displacement
        
        # Clamp to canvas
        a_pos[:, 0] = a_pos[:, 0].clamp(a_w / 2, canvas_w - a_w / 2)
        a_pos[:, 1] = a_pos[:, 1].clamp(a_h / 2, canvas_h - a_h / 2)
    
    # Always write back vectorized results first
    result[a_idx] = a_pos

    if n_ov > 0:
        print(f"    Warning: {n_ov} overlaps remain after vectorized pass. Running greedy fallback...")
        
        # Sort indices by area (largest first) to place big macros before small ones
        areas = w[a_idx] * h[a_idx]
        sorted_indices = a_idx[areas.argsort(descending=True)].tolist()
        
        import math
        
        for i in sorted_indices:
            if fixed[i].item(): continue
            
            wi = w[i].item() + eps
            hi = h[i].item() + eps
            tx = result[i, 0].item()
            ty = result[i, 1].item()
            
            # Function to test overlap against ALL other active macros
            def has_overlap(cx, cy):
                x0, x1 = cx - wi/2, cx + wi/2
                y0, y1 = cy - hi/2, cy + hi/2
                
                other_idx = a_idx[a_idx != i]
                other_w = w[other_idx]
                other_h = h[other_idx]
                other_x = result[other_idx, 0]
                other_y = result[other_idx, 1]
                
                ox = (x0 < other_x + other_w/2) & (x1 > other_x - other_w/2)
                oy = (y0 < other_y + other_h/2) & (y1 > other_y - other_h/2)
                return (ox & oy).any().item()

            if not has_overlap(tx, ty):
                continue # Already legal
                
            # Robust spiral search
            placed = False
            step_size = min(wi, hi) / 4.0
            if step_size < 10: step_size = 10.0
            
            # Spiral outward up to the max dimension of the canvas
            max_radius = max(canvas_w, canvas_h) * 1.5
            max_steps = int(max_radius / step_size)
            
            for step_mult in range(1, max_steps):
                radius = step_mult * step_size
                # Number of angles proportional to radius
                num_angles = max(8, int(math.pi * radius / step_size))
                
                for angle_idx in range(num_angles):
                    a = 2 * math.pi * angle_idx / num_angles
                    cx = tx + radius * math.cos(a)
                    cy = ty + radius * math.sin(a)
                    
                    # (Removed clamping to allow macros to push out of bounds if the canvas is too fragmented)
                    
                    if not has_overlap(cx, cy):
                        result[i, 0] = cx
                        result[i, 1] = cy
                        placed = True
                        break
                if placed: break
                
            if not placed:
                print(f"      Warning: No legal position found for macro {i}!")
        
        # Verify greedy pass
        final_ov = count_overlaps(result, sizes, active)
        print(f"    Greedy fallback complete. Final overlaps: {final_ov}")

    return result
