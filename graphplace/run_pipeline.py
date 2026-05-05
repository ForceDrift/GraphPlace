"""
Consolidated pipeline wrapper for GraphPlace.

Usage (from project root):
    python graphplace/run_pipeline.py --design ibm01 --vis
    python graphplace/run_pipeline.py --design ibm02
    python graphplace/run_pipeline.py --design ibm01 --vis --no-run   # just visualize existing result
"""
import argparse
import sys
import torch
from pathlib import Path

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Import the DREAMPlace runner
sys.path.insert(0, str(PROJECT_ROOT))
from graphplace.dreamplace.run_dreamplace_from_pt import (
    get_paths, load_pt, parse_pb_nets,
    write_nodes, write_nets, write_pl, write_scl, write_wts, write_aux,
    write_json_config, find_result_pl, pl_to_pt,
    DOCKER_CONTAINER, ROOT
)
from graphplace.core.legalizer import legalize, count_overlaps


def main():
    parser = argparse.ArgumentParser(description="GraphPlace Pipeline: DREAMPlace placement + legalization + visualization")
    parser.add_argument("--design", type=str, default="ibm01", help="Benchmark name (e.g. ibm01)")
    parser.add_argument("--vis", action="store_true", help="Generate visualization after placement")
    parser.add_argument("--no-run", action="store_true", help="Skip DREAMPlace run, just visualize existing result")
    parser.add_argument("--pt", type=str, default=None, help="Override input .pt file path")

    args = parser.parse_args()
    design = args.design
    paths = get_paths(design, args.pt)

    if not args.no_run:
        # ── Step 1: Generate Bookshelf + run DREAMPlace ──
        print(f"{'='*60}")
        print(f"  GraphPlace Pipeline: {design}")
        print(f"{'='*60}")

        if not paths["pt_file"].exists():
            print(f"ERROR: {paths['pt_file']} not found.")
            return

        paths["out_dir"].mkdir(parents=True, exist_ok=True)

        # Load .pt
        data = load_pt(paths["pt_file"])

        # Resolve net connectivity
        net_nodes_raw = data.get('net_nodes', [])
        if len(net_nodes_raw) == 0:
            pb_file = ROOT / f"externals/MacroPlacement/Testcases/ICCAD04/{design}/netlist.pb.txt"
            if not pb_file.exists():
                print(f"ERROR: No net_nodes in .pt and no netlist.pb.txt at {pb_file}")
                return
            net_nodes_list = parse_pb_nets(pb_file, data['macro_names'])
        else:
            net_nodes_list = [n.tolist() if hasattr(n, 'tolist') else list(n)
                              for n in net_nodes_raw]

        # Generate all Bookshelf files
        write_nodes(data, paths["out_dir"] / f"{design}.nodes")
        num_nets, valid_nets = write_nets(net_nodes_list, data['macro_names'],
                                          paths["out_dir"] / f"{design}.nets")
        write_pl(data, paths["out_dir"] / f"{design}.pl")
        write_scl(data, paths["out_dir"] / f"{design}.scl")
        write_wts(num_nets, paths["out_dir"] / f"{design}.wts")
        write_aux(paths["out_dir"], design)
        write_json_config(paths["out_dir"], design, paths)

        # Run DREAMPlace in Docker (global placement)
        import subprocess
        print(f"\n[8] Running DREAMPlace in container {DOCKER_CONTAINER} ...")
        cmd = ["docker", "exec", "-w", "/workspace/externals/DREAMPlace/install", DOCKER_CONTAINER,
               "python", "dreamplace/Placer.py",
               f"{paths['bench_dir_docker']}/{design}.json"]
        print("  $", " ".join(cmd))
        subprocess.run(cmd, check=False)

        # Find and convert DREAMPlace output to .pt
        result_pl = find_result_pl(paths["results_root"], design)
        if result_pl:
            pl_to_pt(result_pl, paths["pt_file"], paths["out_pt"])
        else:
            print("ERROR: DREAMPlace produced no output .pl file.")
            return

    # ── Step 2: Post-placement legalization ──
    print(f"\n[10] Running geometric legalization ...")
    result_data = torch.load(str(paths["out_pt"]), weights_only=False)
    pos = result_data['macro_positions']
    sizes = result_data['macro_sizes']
    fixed = result_data['macro_fixed']
    num_hard = result_data['num_hard_macros']
    cw = result_data['canvas_width']
    ch = result_data['canvas_height']

    # Check overlaps before
    n_before = count_overlaps(pos, sizes)
    print(f"    Before legalization: {n_before} total overlaps")

    # Legalize all macros
    legal_pos = legalize(pos, sizes, fixed, cw, ch,
                         hard_only=False,
                         max_iter=2000, eps=0.01)

    # Verify
    hard_mask = torch.zeros(pos.shape[0], dtype=torch.bool)
    hard_mask[:num_hard] = True
    n_hard_after = count_overlaps(legal_pos, sizes, hard_mask)
    n_total_after = count_overlaps(legal_pos, sizes)
    print(f"    After legalization: {n_hard_after} hard overlaps, {n_total_after} total overlaps")

    # Save
    result_data['macro_positions'] = legal_pos
    torch.save(result_data, str(paths["out_pt"]))
    print(f"\nPlacement saved to {paths['out_pt']}")

    # ── Step 3: Visualization ──
    if args.vis:
        out_pt = paths["out_pt"]
        if not out_pt.exists():
            print(f"ERROR: Result file {out_pt} not found. Run without --no-run first.")
            return

        print(f"\n[11] Generating visualization for {design} ...")
        import subprocess
        vis_cmd = [
            sys.executable,
            str(PROJECT_ROOT / "graphplace" / "utils" / "vis" / "visualize_graph_cli.py"),
            "--bench", f"{design}_dreamplace",
            "--data-dir", "data/generated",
            "--out", f"{design}_final.png",
            "--max-edges", "50000"
        ]
        subprocess.run(vis_cmd, cwd=str(PROJECT_ROOT))
        print(f"Visualization saved to {PROJECT_ROOT / f'{design}_final.png'}")


if __name__ == "__main__":
    main()
