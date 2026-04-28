import torch
import numpy as np
from typing import Tuple, List
from models import Benchmark

class Legalizer:
    """
    Fast Greedy Legalizer that preserves placement.
    It groups macros into rows based on original Y and resolves overlaps.
    """
    def __init__(self, benchmark: Benchmark):
        self.benchmark = benchmark
        self.num_macros = benchmark.num_macros
        self.x = benchmark.macro_positions[:, 0].clone().float()
        self.y = benchmark.macro_positions[:, 1].clone().float()
        self.widths = benchmark.macro_sizes[:, 0].float()
        self.heights = benchmark.macro_sizes[:, 1].float()
        self.fixed = benchmark.macro_fixed
        self.canvas_w = float(benchmark.canvas_width)
        self.canvas_h = float(benchmark.canvas_height)

    def legalize(self):
        print(f"Starting Robust Greedy Legalization for {self.benchmark.name}...")
        
        self._snap_to_boundary()
        
        # 1. Resolve overlaps by iterative shifting (Preserving phase)
        for _ in range(50):
            ii, jj = torch.triu_indices(self.num_macros, self.num_macros, offset=1)
            l = self.x - self.widths / 2
            r = self.x + self.widths / 2
            b = self.y - self.heights / 2
            t = self.y + self.heights / 2
            
            ox = (torch.minimum(r[ii], r[jj]) - torch.maximum(l[ii], l[jj])).clamp(min=0)
            oy = (torch.minimum(t[ii], t[jj]) - torch.maximum(b[ii], b[jj])).clamp(min=0)
            
            overlap = (ox > 0) & (oy > 0)
            if not overlap.any():
                break
                
            idx_i = ii[overlap]
            idx_j = jj[overlap]
            ox_val = ox[overlap]
            oy_val = oy[overlap]
            
            # Choose axis of least overlap
            use_x = ox_val < oy_val
            
            # Displacement
            for k in range(len(idx_i)):
                i, j = idx_i[k], idx_j[k]
                if self.fixed[i] and self.fixed[j]: continue
                
                if use_x[k]:
                    shift = (ox_val[k] + 0.1) / 2
                    direction = 1.0 if self.x[i] >= self.x[j] else -1.0
                    if not self.fixed[i]: self.x[i] += direction * shift
                    if not self.fixed[j]: self.x[j] -= direction * shift
                else:
                    shift = (oy_val[k] + 0.1) / 2
                    direction = 1.0 if self.y[i] >= self.y[j] else -1.0
                    if not self.fixed[i]: self.y[i] += direction * shift
                    if not self.fixed[j]: self.y[j] -= direction * shift
            
            self._snap_to_boundary()

        # 2. Final Rescue Pass (Guarantee phase)
        # If still not legal, we use a simple row-packing for the remaining overlapping macros
        is_legal, msg = self.check_legality()
        if not is_legal:
            print(f"Initial passes left some overlaps: {msg}. Applying greedy row-pack rescue...")
            self._greedy_row_pack_rescue()
        
        is_legal, msg = self.check_legality()
        print(f"Legalization {'successful' if is_legal else 'partial: ' + msg}.")
        
        self.benchmark.macro_positions[:, 0] = self.x
        self.benchmark.macro_positions[:, 1] = self.y
        return is_legal

    def _greedy_row_pack_rescue(self):
        """
        Groups macros into rows based on Y and resolves horizontal overlaps.
        This is a robust fallback to ensure legality.
        """
        movable_indices = [i for i in range(self.num_macros) if not self.fixed[i]]
        movable_indices.sort(key=lambda i: self.y[i].item())
        
        # Very simple binning into rows
        rows = []
        if movable_indices:
            current_row = [movable_indices[0]]
            last_y = self.y[movable_indices[0]].item()
            row_h = self.heights[movable_indices[0]].item()
            
            for i in movable_indices[1:]:
                if self.y[i].item() < last_y + row_h / 2:
                    current_row.append(i)
                else:
                    rows.append(current_row)
                    current_row = [i]
                    last_y = self.y[i].item()
                    row_h = self.heights[i].item()
            rows.append(current_row)
            
        last_max_y = 0.0
        for row in rows:
            row.sort(key=lambda i: self.x[i].item())
            max_h = max([self.heights[i].item() for i in row])
            target_y = max(sum([self.y[i].item() for i in row])/len(row), last_max_y + max_h/2)
            
            curr_x = 0.0
            for i in row:
                w = self.widths[i].item()
                self.x[i] = max(self.x[i].item(), curr_x + w/2)
                self.y[i] = target_y
                curr_x = self.x[i].item() + w/2 + 0.1
                
            if curr_x > self.canvas_w:
                # Compress if overflow
                scale = self.canvas_w / curr_x
                new_curr_x = 0.0
                for i in row:
                    w = self.widths[i].item()
                    self.x[i] = new_curr_x + w/2
                    new_curr_x += w + 0.1
            
            last_max_y = target_y + max_h/2

    def _snap_to_boundary(self):
        w2, h2 = self.widths/2, self.heights/2
        self.x = torch.clamp(self.x, w2, self.canvas_w - w2)
        self.y = torch.clamp(self.y, h2, self.canvas_h - h2)

    def check_legality(self) -> Tuple[bool, str]:
        ii, jj = torch.triu_indices(self.num_macros, self.num_macros, offset=1)
        l, r = self.x - self.widths/2, self.x + self.widths/2
        b, t = self.y - self.heights/2, self.y + self.heights/2
        
        ox = (torch.minimum(r[ii], r[jj]) - torch.maximum(l[ii], l[jj])).clamp(min=0)
        oy = (torch.minimum(t[ii], t[jj]) - torch.maximum(b[ii], b[jj])).clamp(min=0)
        overlap = (ox > 0.01) & (oy > 0.01)
        
        if overlap.any():
            idx = torch.where(overlap)[0][0]
            return False, f"Overlap {ii[idx]}-{jj[idx]}"
        return True, "Legal"
