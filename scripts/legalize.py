#!/usr/bin/env python3
"""
Mixed-Size Bookshelf Legalizer (multi-row macros)

For each movable cell:
  1. Snap to nearest legal row-aligned y (cell bottom must be on a row boundary)
  2. For the chosen y-band, try to place at GP x with no overlap against all already-placed cells
  3. Use Tetris-greedy: sort cells by GP x, resolve conflicts with left/right probing

Usage:
    python3 scripts/legalize.py \\
        --nodes  data/ibm01_bookshelf/ibm01.nodes \\
        --scl    data/ibm01_bookshelf/ibm01.scl \\
        --input  externals/RePlAce/build/output/bookshelf/ibm01/experiment011/ibm01.eplace-gp.pl \\
        --output output/ibm01/ibm01_legalized.pl
"""

import argparse
import bisect
import math
from collections import defaultdict


# ─────────────────────────────  Parsers  ──────────────────────────────────

def parse_nodes(nodes_file):
    sizes = {}
    fixed_nodes = set()
    with open(nodes_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('UCLA'):
                continue
            if line.startswith(('NumNodes', 'NumTerminals')):
                continue
            parts = line.split()
            if len(parts) >= 3:
                name = parts[0]
                try:
                    w, h = float(parts[1]), float(parts[2])
                    sizes[name] = (w, h)
                except ValueError:
                    continue
                if 'terminal' in parts[3:]:
                    fixed_nodes.add(name)
    return sizes, fixed_nodes


def parse_scl(scl_file):
    """Return sorted list of (y_coord, height, x_start, x_end, site_spacing)."""
    rows = []
    current = {}
    with open(scl_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('UCLA'):
                continue
            if line.startswith('CoreRow'):
                current = {}
            elif line.startswith('End'):
                if current:
                    coord    = current.get('coord', 0.0)
                    h        = current.get('height', 0.0)
                    origin   = current.get('origin', 0.0)
                    numsites = current.get('numsites', 0)
                    site_sp  = current.get('sitespacing', 1.0)
                    x_end    = origin + numsites * site_sp
                    rows.append((coord, h, origin, x_end, site_sp))
                    current = {}
            else:
                kv = line.replace(':', ' ').split()
                if not kv:
                    continue
                k = kv[0].lower()
                if k == 'coordinate'    and len(kv) > 1: current['coord']       = float(kv[1])
                elif k == 'height'      and len(kv) > 1: current['height']      = float(kv[1])
                elif k == 'sitespacing' and len(kv) > 1: current['sitespacing'] = float(kv[1])
                elif k == 'subroworigin':
                    if len(kv) > 1: current['origin'] = float(kv[1])
                    try:
                        ni = [s.lower() for s in kv].index('numsites')
                        current['numsites'] = int(kv[ni + 1])
                    except (ValueError, IndexError):
                        pass
                elif k == 'numsites'   and len(kv) > 1: current['numsites']    = int(kv[1])
    rows.sort()
    return rows


def parse_pl(pl_file):
    placement = {}
    with open(pl_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('UCLA'):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            name = parts[0]
            try:
                x, y = float(parts[1]), float(parts[2])
            except ValueError:
                continue
            is_fixed = '/FIXED' in line or '/FIXED_NI' in line
            orient = 'N'
            if ':' in parts:
                ci = parts.index(':')
                if ci + 1 < len(parts):
                    orient = parts[ci + 1]
            placement[name] = (x, y, orient, is_fixed)
    return placement


# ─────────────────────────────  Overlap helper  ────────────────────────────

class PlacedCell:
    __slots__ = ('name', 'x', 'y', 'w', 'h')
    def __init__(self, name, x, y, w, h):
        self.name = name
        self.x = x
        self.y = y
        self.w = w
        self.h = h


def intervals_overlap(ax, aw, bx, bw, eps=1e-9):
    return ax < bx + bw - eps and bx < ax + aw - eps


def find_free_x(placed_cells, x_want, y, w, h, x_min, x_max, eps=1e-9):
    """
    Find the x position closest to x_want for a cell of size (w,h)
    placed at row y, that doesn't overlap any cell in placed_cells.
    Searches outward from x_want.  Returns None if no room.
    """
    # Collect cells that overlap in y with [y, y+h)
    blocking = []
    for pc in placed_cells:
        if intervals_overlap(y, h, pc.y, pc.h, eps):
            blocking.append((pc.x, pc.w))

    # Build sorted list of occupied x-intervals in this y-band
    blocking.sort()

    # Try x_want first
    def is_free(x):
        if x < x_min - eps or x + w > x_max + eps:
            return False
        for bx, bw in blocking:
            if intervals_overlap(x, w, bx, bw, eps):
                return False
        return True

    if is_free(x_want):
        return x_want

    # Build candidate positions = gaps between blockers
    candidates = [x_min]
    for bx, bw in blocking:
        candidates.append(bx - w)   # left of blocker
        candidates.append(bx + bw)  # right of blocker
    candidates.append(x_max - w)

    best_x = None
    best_cost = float('inf')
    for cx in candidates:
        cx = max(x_min, min(cx, x_max - w))
        if is_free(cx):
            cost = abs(cx - x_want)
            if cost < best_cost:
                best_cost = cost
                best_x = cx
    return best_x


# ─────────────────────────────  Main Legalizer  ────────────────────────────

def legalize(nodes_file, scl_file, input_pl, output_pl):
    print("Parsing nodes ...")
    sizes, fixed_nodes = parse_nodes(nodes_file)
    print(f"  {len(sizes)} nodes, {len(fixed_nodes)} terminals")

    print("Parsing SCL rows ...")
    rows = parse_scl(scl_file)
    print(f"  {len(rows)} placement rows")
    if not rows:
        raise RuntimeError("No rows found in .scl")

    row_ys = [r[0] for r in rows]
    row_info = { r[0]: r for r in rows }     # y -> row tuple

    canvas_xmin = min(r[2] for r in rows)
    canvas_xmax = max(r[3] for r in rows)
    canvas_ymin = rows[0][0]
    canvas_ymax  = rows[-1][0] + rows[-1][1]
    row_h = rows[0][1]
    print(f"  Canvas x=[{canvas_xmin:.3f},{canvas_xmax:.3f}] y=[{canvas_ymin:.3f},{canvas_ymax:.3f}]  row_h={row_h:.5f}")

    print("Parsing GP placement ...")
    placement = parse_pl(input_pl)
    print(f"  {len(placement)} entries")

    # ── Place fixed cells first ──
    placed = []        # list of PlacedCell (fixed + done movable)
    out_pos = {}       # name -> (x, y, orient, is_fixed)

    for name, (x, y, orient, is_fixed) in placement.items():
        if is_fixed:
            w, h = sizes.get(name, (0, 0))
            placed.append(PlacedCell(name, x, y, w, h))
            out_pos[name] = (x, y, orient, True)

    # ── Sort movable by GP x for Tetris ordering ──
    movable = [
        (x, name, orient)
        for name, (x, y, orient, is_fixed) in placement.items()
        if not is_fixed and name in sizes
    ]
    movable.sort(key=lambda t: t[0])

    total_disp = 0.0
    skipped = 0

    for gp_x, name, orient in movable:
        gp_y = placement[name][1]
        w, h = sizes[name]

        # ── 1. Find the best legal row y for this cell ──
        # Cell can span multiple rows as long as bottom-left is on a row boundary.
        # Try rows closest to gp_y first.
        n_rows_needed = max(1, round(h / row_h))

        # legal y positions: row boundaries where cell fits within canvas height
        legal_ys = []
        for ry in row_ys:
            if ry + h <= canvas_ymax + 1e-6:
                legal_ys.append(ry)

        if not legal_ys:
            print(f"  WARNING: no valid row y for {name} (h={h:.4f}), skipping")
            skipped += 1
            continue

        # Sort by distance from GP y
        legal_ys.sort(key=lambda ry: abs(ry - gp_y))

        placed_y = None
        placed_x = None

        for try_y in legal_ys[:8]:   # try top-8 closest rows
            free_x = find_free_x(placed, gp_x, try_y, w, h, canvas_xmin, canvas_xmax)
            if free_x is not None:
                # Choose the y closest to GP that has room
                if placed_y is None or abs(try_y - gp_y) < abs(placed_y - gp_y):
                    placed_y = try_y
                    placed_x = free_x
                break   # first valid row (sorted by distance) is best

        if placed_y is None:
            # Last resort: canvas left, nearest row
            placed_y = legal_ys[0]
            placed_x = canvas_xmin
            print(f"  WARN: forced {name} to ({placed_x},{placed_y})")

        disp = math.sqrt((placed_x - gp_x)**2 + (placed_y - gp_y)**2)
        total_disp += disp

        placed.append(PlacedCell(name, placed_x, placed_y, w, h))
        out_pos[name] = (placed_x, placed_y, orient, False)

    avg_disp = total_disp / max(len(movable) - skipped, 1)
    print(f"\nLegalization done:")
    print(f"  Placed: {len(movable) - skipped}, Skipped: {skipped}")
    print(f"  Avg displacement: {avg_disp:.4f}")

    # ── Write output .pl ──
    import os
    os.makedirs(os.path.dirname(os.path.abspath(output_pl)), exist_ok=True)
    with open(output_pl, 'w') as f:
        f.write("UCLA pl 1.0 \n")
        f.write("# Legalized placement (multi-row Tetris) from RePlAce GP\n\n")
        for name, (x, y, orient, is_fixed) in sorted(out_pos.items()):
            suffix = " /FIXED" if is_fixed else ""
            f.write(f"{name}\t{x:.6f}\t{y:.6f}\t: {orient}{suffix}\n")

    print(f"Wrote: {output_pl}")


def main():
    ap = argparse.ArgumentParser(description="Multi-row Bookshelf Legalizer")
    ap.add_argument('--nodes',  required=True)
    ap.add_argument('--scl',    required=True)
    ap.add_argument('--input',  required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    legalize(args.nodes, args.scl, args.input, args.output)


if __name__ == '__main__':
    main()
