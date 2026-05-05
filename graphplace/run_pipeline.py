import argparse
import sys
import torch
import os
from pathlib import Path

# Add project root and challenge submission directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHALLENGE_ROOT = PROJECT_ROOT / "externals" / "macro-place-challenge-2026"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CHALLENGE_ROOT))

# Now we can import the competition logic and our visualization
from submissions.dreamplace_eval import DREAMPlacePlacer
from macro_place.loader import load_benchmark
from graphplace.utils.vis.vis import plot_graph
from types import SimpleNamespace

def main():
    parser = argparse.ArgumentParser(description="Consolidated GraphPlace Pipeline Wrapper")
    parser.add_argument("--design", type=str, default="ibm01", help="Benchmark name (e.g. ibm01)")
    parser.add_argument("--vis", action="store_true", help="Flag to automatically generate visualization after placement")
    
    args = parser.parse_args()
    design = args.design
    
    print(f"=== Starting Pipeline for {design} ===")
    
    # 1. Run Evaluation (which runs DREAMPlace + Legalization + Saves result)
    print(f"Running DREAMPlace + Full Legalization via Challenge Harness...")
    import subprocess
    cmd = [
        "uv", "run", "evaluate", 
        "submissions/dreamplace_eval.py", 
        "--benchmark", design
    ]
    # We MUST run this from the challenge root for the harness to work
    proc = subprocess.run(cmd, cwd=str(CHALLENGE_ROOT), check=False)
    
    if proc.returncode != 0:
        print(f"Error: Evaluation failed with return code {proc.returncode}")
        return

    # 2. Visualization
    if args.vis:
        print(f"Generating visualization for {design}...")
        out_pt = PROJECT_ROOT / "data" / "generated" / f"{design}_dreamplace.pt"
        
        if not out_pt.exists():
            print(f"Error: Legalized result not found at {out_pt}. Did evaluate save it?")
            return

        # Use the CLI tool we already built
        vis_cmd = [
            "python", "graphplace/utils/vis/visualize_graph_cli.py",
            "--bench", f"{design}_dreamplace",
            "--data-dir", "data/generated",
            "--out", f"{design}_final.png",
            "--max-edges", "50000"
        ]
        subprocess.run(vis_cmd, cwd=str(PROJECT_ROOT))
        print(f"Visualization saved to {PROJECT_ROOT / (design + '_final.png')}")

    print(f"=== Pipeline Finished for {design} ===")

if __name__ == "__main__":
    main()
