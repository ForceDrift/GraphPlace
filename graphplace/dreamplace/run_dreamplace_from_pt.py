"""
Pipeline: .pt file → Bookshelf format → DREAMPlace → output .pt

Usage:
    python graphplace/dreamplace/run_dreamplace_from_pt.py --design ibm02 --run
    python graphplace/dreamplace/run_dreamplace_from_pt.py --design ibm01 --run

The .pt files in data/processed/public/ contain ALL information needed:
  - macro_names, macro_sizes, macro_positions, macro_fixed
  - canvas_width, canvas_height, grid_rows, grid_cols
  - net_nodes (connectivity), net_weights
  - hroutes_per_micron, vroutes_per_micron

This script generates every Bookshelf file (.nodes, .nets, .pl, .scl, .wts, .aux)
directly from the .pt, then runs DREAMPlace in the Docker container.
"""
import re
import json
import argparse
import subprocess
import torch
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────────
ROOT             = Path(__file__).resolve().parent.parent.parent   # GraphPlace/
DOCKER_CONTAINER = "dreamplace_container"   # update if container is restarted

# DREAMPlace Bookshelf unit: 1 unit = 0.001 micron  (1 µm = 1000 units)
UNIT = 1000


def get_paths(design: str, pt_override: str = None):
    """Return all design-specific paths."""
    return {
        "pt_file":           Path(pt_override) if pt_override
                             else ROOT / f"data/processed/public/{design}.pt",
        "out_dir":           ROOT / f"externals/DREAMPlace/install/benchmarks/{design}_dreamplace",
        "results_root":      ROOT / f"externals/DREAMPlace/install/results/{design}",
        "out_pt":            ROOT / f"data/generated/{design}_dreamplace.pt",
        "bench_dir_docker":  f"benchmarks/{design}_dreamplace",
        "result_dir_docker": f"results/{design}",
    }


# ── Load .pt ───────────────────────────────────────────────────────────────────
def load_pt(pt_file: Path):
    print(f"[0] Loading {pt_file} ...")
    data = torch.load(pt_file, weights_only=False)
    print(f"    {data['name']}: {data['num_macros']} macros "
          f"({data['num_hard_macros']} hard, {data['num_soft_macros']} soft), "
          f"{data['num_nets']} nets, "
          f"canvas {data['canvas_width']:.2f} x {data['canvas_height']:.2f} um")
    return data


# ── Parse netlist.pb.txt for net connectivity (fallback) ──────────────────────
def parse_pb_nets(pb_file: Path, macro_names: list):
    """
    Parse the protobuf netlist to extract nets as lists of macro indices.
    Only includes nets where ≥2 nodes appear in macro_names.
    Returns list of lists of macro indices.
    """
    print(f"    Parsing connectivity from {pb_file} ...")
    name_to_idx = {name: i for i, name in enumerate(macro_names)}

    all_node_names = []
    all_nets = []       # list of (sink_name, [input_names])
    in_node = False
    current_name = None
    inputs = []

    with open(pb_file, 'r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if line == 'node {':
                in_node = True
                current_name = None
                inputs = []
            elif line == '}' and in_node:
                if current_name and current_name != '__metadata__':
                    all_node_names.append(current_name)
                    if inputs:
                        all_nets.append((current_name, inputs[:]))
                in_node = False
            elif in_node:
                m = re.match(r'name:\s*"(.*)"', line)
                if m: current_name = m.group(1)
                m = re.match(r'input:\s*"(.*)"', line)
                if m: inputs.append(m.group(1))

    # Build global name→idx for pb nodes
    pb_name_to_idx = {name: i for i, name in enumerate(all_node_names)}

    # Convert to macro-index nets
    net_nodes = []
    for sink, input_names in all_nets:
        macro_indices = []
        # sink itself
        if sink in name_to_idx:
            macro_indices.append(name_to_idx[sink])
        # each input source (strip pin suffix like "foo/IP1" → "foo")
        for src in input_names:
            src_node = src.split('/')[0]
            if src_node in name_to_idx and src_node != sink:
                idx = name_to_idx[src_node]
                if idx not in macro_indices:
                    macro_indices.append(idx)
        if len(macro_indices) >= 2:
            net_nodes.append(macro_indices)

    print(f"    Parsed {len(net_nodes)} nets connecting macros from protobuf")
    return net_nodes


def write_nodes(data: dict, out_path: Path):
    """Write Bookshelf .nodes from .pt macro_names + macro_sizes + macro_fixed."""
    print(f"[1] Writing {out_path} ...")
    names   = data['macro_names']
    sizes   = data['macro_sizes']    # [N, 2] in microns
    fixed   = data['macro_fixed']    # [N] bool
    n       = data['num_macros']

    num_terminals = int(fixed.sum().item())

    with open(out_path, 'w') as f:
        f.write("UCLA nodes 1.0\n\n")
        f.write(f"NumNodes : {n}\n")
        f.write(f"NumTerminals : {num_terminals}\n")
        for i, name in enumerate(names):
            w = int(round(sizes[i, 0].item() * UNIT))
            h = int(round(sizes[i, 1].item() * UNIT))
            if fixed[i].item():
                f.write(f"\t{name}\t{w}\t{h}\tterminal\n")
            else:
                f.write(f"\t{name}\t{w}\t{h}\n")

    print(f"    Wrote {n} nodes ({num_terminals} terminals)")


# ── Generate .nets ─────────────────────────────────────────────────────────────
def write_nets(net_nodes: list, macro_names: list, out_path: Path):
    """Write Bookshelf .nets from a list of nets (each net = list of macro indices)."""
    print(f"[2] Writing {out_path} ...")
    valid_nets = [n for n in net_nodes if len(n) >= 2]
    num_nets = len(valid_nets)
    num_pins = sum(len(n) for n in valid_nets)

    with open(out_path, 'w') as f:
        f.write("UCLA nets 1.0\n\n")
        f.write(f"NumNets : {num_nets}\n")
        f.write(f"NumPins : {num_pins}\n\n")

        for net_idx, net in enumerate(valid_nets):
            degree = len(net)
            f.write(f"NetDegree : {degree}   net{net_idx}\n")
            for pin_i, node_idx in enumerate(net):
                node_name = macro_names[node_idx.item() if hasattr(node_idx, 'item') else node_idx]
                io = 'O' if pin_i == 0 else 'I'
                f.write(f"\t{node_name}\t{io} : 0.0\t0.0\n")
            f.write("\n")

    print(f"    Wrote {num_nets} nets, {num_pins} pins")
    return num_nets, valid_nets


# ── Generate .pl ───────────────────────────────────────────────────────────────
def write_pl(data: dict, out_path: Path):
    """Write Bookshelf .pl from .pt macro_positions.
    
    Bookshelf .pl uses BOTTOM-LEFT corner coordinates.
    Our .pt stores CENTER coordinates, so we convert: BL = center - size/2
    """
    print(f"[3] Writing {out_path} ...")
    names     = data['macro_names']
    positions = data['macro_positions']   # [N, 2] center coords in microns
    sizes     = data['macro_sizes']       # [N, 2] (w, h) in microns
    fixed     = data['macro_fixed']

    with open(out_path, 'w') as f:
        f.write("UCLA pl 1.0\n\n")
        for i, name in enumerate(names):
            # Convert center → bottom-left
            bl_x = positions[i, 0].item() - sizes[i, 0].item() / 2
            bl_y = positions[i, 1].item() - sizes[i, 1].item() / 2
            x = int(round(bl_x * UNIT))
            y = int(round(bl_y * UNIT))
            orient = "N"
            fixed_flag = " /FIXED" if fixed[i].item() else ""
            f.write(f"\t{name}\t{x}\t{y}\t: {orient}{fixed_flag}\n")

    print(f"    Wrote {len(names)} placements (bottom-left coords)")


# ── Generate .scl ──────────────────────────────────────────────────────────────
def write_scl(data: dict, out_path: Path):
    """Write Bookshelf .scl (row info) derived from canvas dimensions in .pt."""
    print(f"[4] Writing {out_path} ...")
    canvas_w = data['canvas_width']   # microns
    canvas_h = data['canvas_height']  # microns

    # Row height: use standard cell row height = 1µm → 1000 units
    row_height  = 1000   # units
    canvas_w_u  = int(round(canvas_w * UNIT))
    canvas_h_u  = int(round(canvas_h * UNIT))
    num_rows    = canvas_h_u // row_height
    num_sites   = canvas_w_u  # 1 site = 1 unit wide

    with open(out_path, 'w') as f:
        f.write("UCLA scl 1.0\n\n")
        f.write(f"NumRows : {num_rows}\n\n")
        for row in range(num_rows):
            coord = row * row_height
            f.write("CoreRow Horizontal\n")
            f.write(f"  Coordinate    : {coord}\n")
            f.write(f"  Height        : {row_height}\n")
            f.write(f"  Sitewidth     : 1\n")
            f.write(f"  Sitespacing   : 1\n")
            f.write(f"  Siteorient    : 1\n")
            f.write(f"  Sitesymmetry  : 1\n")
            f.write(f"  SubrowOrigin  : 0\tNumSites : {num_sites}\n")
            f.write("End\n")

    print(f"    Wrote {num_rows} rows, canvas {canvas_w_u} x {canvas_h_u} units")


# ── Generate .wts ──────────────────────────────────────────────────────────────
def write_wts(num_nets: int, out_path: Path):
    """Write a uniform-weight .wts file (weight=1.0 for all nets)."""
    print(f"[5] Writing {out_path} ...")
    with open(out_path, 'w') as f:
        f.write("UCLA wts 1.0\n\n")
        for i in range(num_nets):
            f.write(f"  net{i}  1.000\n")
    print(f"    Wrote {num_nets} net weights")


# ── Generate .aux ──────────────────────────────────────────────────────────────
def write_aux(out_dir: Path, design: str):
    aux = out_dir / f"{design}.aux"
    with open(aux, 'w') as f:
        f.write(f"RowBasedPlacement : {design}.nodes {design}.nets "
                f"{design}.wts {design}.pl {design}.scl\n")
    print(f"[6] Wrote {aux}")


# ── Generate DREAMPlace JSON config ───────────────────────────────────────────
def write_json_config(out_dir: Path, design: str, paths: dict):
    cfg = {
        "aux_input":  f"{paths['bench_dir_docker']}/{design}.aux",
        "gpu": 0,
        "num_bins_x": 512,
        "num_bins_y": 512,
        "global_place_stages": [{
            "num_bins_x": 512, "num_bins_y": 512,
            "iteration": 1000, "learning_rate": 0.01,
            "wirelength": "weighted_average", "optimizer": "nesterov",
            "Llambda_density_weight_iteration": 1, "Lsub_iteration": 1
        }],
        "target_density": 1.0,
        "density_weight": 8e-5,
        "gamma": 4.0,
        "random_seed": 1000,
        "scale_factor": 1.0,
        "ignore_net_degree": 100,
        "enable_fillers": 1,
        "gp_noise_ratio": 0.025,
        "global_place_flag": 1,
        "legalize_flag": 0,
        "abacus_legalize_flag": 0,
        "detailed_place_flag": 0,
        "detailed_place_engine": "",
        "detailed_place_command": "",
        "stop_overflow": 0.07,
        "dtype": "float32",
        "plot_flag": 0,
        "random_center_init_flag": 0,
        "sort_nets_by_degree": 0,
        "num_threads": 8,
        "deterministic_flag": 1,
        "result_dir": paths["result_dir_docker"],
    }
    cfg_path = out_dir / f"{design}.json"
    with open(cfg_path, 'w') as f:
        json.dump(cfg, f, indent=4)
    print(f"[7] Wrote {cfg_path}")
    return cfg_path


# ── Convert output .pl back to .pt ─────────────────────────────────────────────
def find_result_pl(results_root: Path, design: str) -> Path:
    """Dynamically find the .gp.pl (or .pl) written by DREAMPlace under results_root."""
    # DREAMPlace may write to results/<design>/<design>.gp.pl OR
    # results/<design>/<design>/<design>.gp.pl depending on version
    candidates = list(results_root.rglob("*.gp.pl")) + list(results_root.rglob(f"{design}.pl"))
    if candidates:
        # prefer the most recently modified
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]
    return None


def pl_to_pt(pl_file: Path, original_pt_file: Path, out_pt_file: Path):
    """Convert DREAMPlace output .pl back to .pt.
    
    DREAMPlace .pl uses bottom-left corners in DREAMPlace units.
    We convert back to center coordinates in microns.
    """
    print(f"[9] Converting placement {pl_file} to {out_pt_file} ...")
    if not pl_file or not pl_file.exists():
        print(f"    ERROR: Placement file not found: {pl_file}")
        return

    data = torch.load(original_pt_file, weights_only=False)
    macro_names     = data['macro_names']
    macro_sizes     = data['macro_sizes']  # [N, 2] in microns
    name_to_idx     = {name: i for i, name in enumerate(macro_names)}
    macro_positions = data['macro_positions'].clone()

    updated = 0
    with open(pl_file, 'r') as f:
        for line in f:
            if line.startswith('UCLA') or line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 4:
                name = parts[0]
                if name in name_to_idx:
                    idx = name_to_idx[name]
                    # BL corner in DREAMPlace units → center in microns
                    bl_x_um = float(parts[1]) / UNIT
                    bl_y_um = float(parts[2]) / UNIT
                    w_um = macro_sizes[idx, 0].item()
                    h_um = macro_sizes[idx, 1].item()
                    macro_positions[idx, 0] = bl_x_um + w_um / 2
                    macro_positions[idx, 1] = bl_y_um + h_um / 2
                    updated += 1

    data['macro_positions'] = macro_positions
    out_pt_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, out_pt_file)
    print(f"    Updated {updated}/{len(macro_names)} positions, saved to {out_pt_file}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Convert .pt → Bookshelf → DREAMPlace → output .pt"
    )
    ap.add_argument('--design', default='ibm01',
                    help='Benchmark name (ibm01, ibm02, ...) — must exist in data/processed/public/')
    ap.add_argument('--pt',    default=None,
                    help='Override path to input .pt file')
    ap.add_argument('--run',   action='store_true',
                    help='Run DREAMPlace in Docker after generating Bookshelf files')
    args = ap.parse_args()

    design = args.design
    paths  = get_paths(design, args.pt)
    out_dir = paths["out_dir"]

    # Validate
    if not paths["pt_file"].exists():
        print(f"ERROR: {paths['pt_file']} not found.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load
    data = load_pt(paths["pt_file"])

    # Resolve net connectivity
    net_nodes_raw = data.get('net_nodes', [])
    if len(net_nodes_raw) == 0:
        # .pt has no stored nets — parse from netlist.pb.txt
        pb_file = ROOT / f"externals/MacroPlacement/Testcases/ICCAD04/{design}/netlist.pb.txt"
        if not pb_file.exists():
            print(f"ERROR: No net_nodes in .pt and no netlist.pb.txt at {pb_file}")
            return
        net_nodes_list = parse_pb_nets(pb_file, data['macro_names'])
    else:
        net_nodes_list = [n.tolist() if hasattr(n, 'tolist') else list(n)
                         for n in net_nodes_raw]

    # Generate Bookshelf files entirely from .pt
    write_nodes(data, out_dir / f"{design}.nodes")
    num_nets, valid_nets = write_nets(net_nodes_list, data['macro_names'], out_dir / f"{design}.nets")
    write_pl(data, out_dir / f"{design}.pl")
    write_scl(data, out_dir / f"{design}.scl")
    write_wts(num_nets, out_dir / f"{design}.wts")
    write_aux(out_dir, design)
    write_json_config(out_dir, design, paths)

    print(f"\nDone! Bookshelf workspace: {out_dir}")
    print(f"Docker command: python dreamplace/Placer.py {paths['bench_dir_docker']}/{design}.json\n")

    # Run DREAMPlace in Docker
    if args.run:
        print(f"[8] Running DREAMPlace in container {DOCKER_CONTAINER} ...")
        cmd = ["docker", "exec", "-w", "/workspace/externals/DREAMPlace/install", DOCKER_CONTAINER,
               "python", "dreamplace/Placer.py",
               f"{paths['bench_dir_docker']}/{design}.json"]
        print("  $", " ".join(cmd))
        result = subprocess.run(cmd, check=False)

        # Find the output .pl written by DREAMPlace (path varies by version)
        result_pl = find_result_pl(paths["results_root"], design)
        pl_to_pt(result_pl, paths["pt_file"], paths["out_pt"])


if __name__ == "__main__":
    main()
