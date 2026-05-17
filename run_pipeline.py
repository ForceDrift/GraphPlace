import argparse
import subprocess
import sys
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="End-to-end GraphPlace Pipeline")
    parser.add_argument("--train-benchmarks", nargs='+', default=["ibm01", "ibm02", "ibm03", "ibm04"], help="Benchmarks to train on")
    parser.add_argument("--eval-benchmarks", nargs='+', default=["ibm01", "ibm02", "ibm03", "ibm04"], help="Benchmarks to evaluate on")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--steps", type=int, default=50, help="Steps per epoch")
    parser.add_argument("--eval-only", action="store_true", help="Skip training, only run evaluation")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent

    # 1. Training
    if not args.eval_only:
        print("=" * 60)
        print("STARTING TRAINING PIPELINE")
        print("=" * 60)
        train_cmd = [
            sys.executable, "-m", "graphplace.train.train_rl_multi",
            "--benchmarks"
        ] + args.train_benchmarks + [
            "--epochs", str(args.epochs),
            "--steps_per_epoch", str(args.steps)
        ]
        
        print(f"Running: {' '.join(train_cmd)}")
        try:
            subprocess.run(train_cmd, check=True, cwd=project_root)
        except subprocess.CalledProcessError as e:
            print(f"Error: Training failed: {e}")
            sys.exit(1)
        print("Success: Training completed successfully!\n")

    # 2. Evaluation
    print("=" * 60)
    print("STARTING EVALUATION PIPELINE")
    print("=" * 60)
    
    eval_script = project_root / "externals" / "macro-place-challenge-2026" / "macro_place" / "evaluate.py"
    submission_script = project_root / "submissions" / "gnn_placer_submission.py"

    if not eval_script.exists():
        print(f"Error: Cannot find official evaluator at {eval_script}")
        print("Make sure you have cloned the macro-place-challenge-2026 repository.")
        sys.exit(1)

    for bench in args.eval_benchmarks:
        print(f"--- Evaluating {bench} ---")
        eval_cmd = [
            sys.executable, str(eval_script),
            str(submission_script),
            "-b", bench
        ]
        
        env = os.environ.copy()
        env["PYTHONPATH"] = str(project_root / "externals" / "macro-place-challenge-2026") + os.pathsep + env.get("PYTHONPATH", "")
        
        print(f"Running: {' '.join(eval_cmd)}")
        try:
            subprocess.run(eval_cmd, check=True, cwd=project_root, env=env)
        except subprocess.CalledProcessError as e:
            print(f"Error: Evaluation failed for {bench}: {e}")
        print()

    print("Pipeline finished!")

if __name__ == "__main__":
    main()
