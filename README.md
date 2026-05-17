# GraphPlace: GNN-based Macro Placement

GraphPlace is a comprehensive pipeline for VLSI macro placement, integrating state-of-the-art global placement engines (RePlAce, DREAMPlace) with Graph Neural Networks for optimization and legalization.

## 📦 Python Environment Setup

To get the repository running locally or on a server, set up the Python environment:

```bash
# Clone repo
git clone git@github.com:ForceDrift/GraphPlace.git
cd GraphPlace

# Setup Python Virtual Environment
python3 -m venv .venv
source .venv/bin/activate

# Install required dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric gymnasium tqdm absl-py protobuf pandas scipy pyyaml matplotlib
```

---

## 🤖 GNN Reinforcement Learning Training

We use Reinforcement Learning to teach a Graph Neural Network (GNN) to resolve macro overlaps in a continuous space, starting from a high-quality "warm start" baseline.

### 1. Training on Multiple Benchmarks
To train the GNN across all available ICCAD04 benchmarks simultaneously (this will take time, best run via `tmux`):
```bash
source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Start training
python run_pipeline.py \
  --train-benchmarks ibm01 ibm02 ibm03 ibm04 ibm06 ibm07 ibm08 ibm09 ibm10 ibm11 ibm12 ibm13 ibm14 ibm15 ibm16 ibm17 ibm18 \
  --epochs 3600 \
  --steps 10
```
*Note: We skip `ibm05` because it is excluded/missing from the challenge dataset.*

### 2. Evaluating the GNN (Inference)
To run inference on the trained model without retraining, simply use the `--eval-only` flag:
```bash
python run_pipeline.py \
  --eval-benchmarks ibm01 ibm02 ibm03 ibm04 ibm06 ibm07 \
  --eval-only
```
This will automatically load `models/gnn_placer_universal_best.pth` and calculate the proxy costs against the competition harness.

---

## 🚀 RePlAce Setup (macOS ARM64 / Local)

If you are running the legacy RePlAce pipelines locally on Apple Silicon (ARM64), follow these steps:

### 1. Compile RePlAce
Requires: CMake, Bison, Flex, `libboost`.
```bash
cd externals/RePlAce
mkdir -p build && cd build
cmake ..
make -j8
```

### 2. Running ibm01 Benchmark
```bash
./replace -bmflag bookshelf \
  -aux /Users/roshaniruku/code/GraphPlace/data/ibm01_bookshelf/ibm01.aux \
  -den 1.0 -output ./output -onlyGP
```

### 3. Legalization & Scoring
We use a custom legalized designed for the **Macro Placement Challenge 2026** proxy cost metric.
```bash
python3 scripts/legalize_challenge.py \
  --pl externals/RePlAce/build/output/bookshelf/ibm01/experiment011/ibm01.eplace-gp.pl \
  --benchmark ibm01 \
  --output output/ibm01/ibm01_legalized.pt
```

## 📂 Repository Structure
*   `graphplace/`: Core GNN logic, reinforcement learning environments, and graph converters.
*   `scripts/`: Utilities, evaluators, and standalone scripts.
*   `externals/`: Native engines (RePlAce) and the Macro Place Challenge 2026 evaluation harness.
*   `data/`: Benchmark netlists and generated PyG datasets.
*   `models/`: Saved `*.pth` checkpoints for the GNN models.
*   `submissions/`: Inference wrappers used by the official challenge evaluation script.
