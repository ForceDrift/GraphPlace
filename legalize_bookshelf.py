import torch
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from graphplace.core.legalizer import legalize, count_overlaps

def parse_nodes(filename):
    nodes = []
    names = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("NumNodes") or line.startswith("NumTerminals") or line.startswith("UCLA") or line.startswith("#") or not line:
                continue
            parts = line.split()
            if len(parts) < 3: continue
            names.append(parts[0])
            nodes.append([float(parts[1]), float(parts[2])])
    return names, torch.tensor(nodes)

def parse_pl(filename, names):
    name_to_idx = {name: i for i, name in enumerate(names)}
    pos = torch.zeros((len(names), 2))
    fixed = torch.zeros(len(names), dtype=torch.bool)
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("UCLA") or line.startswith("#") or not line:
                continue
            parts = line.split()
            if len(parts) < 3: continue
            name = parts[0]
            if name in name_to_idx:
                idx = name_to_idx[name]
                bl_x = float(parts[1])
                bl_y = float(parts[2])
                pos[idx, 0] = bl_x
                pos[idx, 1] = bl_y
                if "/FIXED" in line or "FIXED" in line:
                    fixed[idx] = True
    return pos, fixed

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", required=True)
    parser.add_argument("--pl", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cw", type=float, required=True)
    parser.add_argument("--ch", type=float, required=True)
    parser.add_argument("--num-hard", type=int, default=None)
    args = parser.parse_args()

    names, sizes = parse_nodes(args.nodes)
    bl_pos, fixed = parse_pl(args.pl, names)
    
    # Convert BL to center for legalization
    pos = bl_pos + sizes / 2
    
    print(f"Checking overlaps for {len(names)} nodes...")
    n_ov = count_overlaps(pos, sizes)
    print(f"Initial total overlaps: {n_ov}")
    
    if n_ov > 0 or args.num_hard:
        print("Legalizing...")
        hard_only = args.num_hard is not None
        legal_pos = legalize(pos, sizes, fixed, args.cw, args.ch, 
                            hard_only=hard_only, num_hard=args.num_hard)
        
        print("Writing legalized PL...")
        with open(args.out, 'w') as f:
            f.write("UCLA pl 1.0\n\n")
            for i, name in enumerate(names):
                new_bl_x = legal_pos[i, 0] - sizes[i, 0] / 2
                new_bl_y = legal_pos[i, 1] - sizes[i, 1] / 2
                f.write(f"{name}\t{new_bl_x:.2f}\t{new_bl_y:.2f}\t: N\n")
        print(f"Done! Legalized PL saved to {args.out}")
    else:
        print("No overlaps found. Copying input to output...")
        import shutil
        shutil.copy(args.pl, args.out)

if __name__ == "__main__":
    main()
