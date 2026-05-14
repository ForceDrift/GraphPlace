# GraphPlace: GNN-based Macro Placement

GraphPlace is a comprehensive pipeline for VLSI macro placement, integrating state-of-the-art global placement engines (RePlAce, DREAMPlace) with Graph Neural Networks for optimization and legalization.

## 🚀 RePlAce Setup (macOS ARM64)

We have ported and optimized RePlAce for native Apple Silicon (ARM64) support.

### 1. Requirements
*   CMake
*   Bison / Flex
*   `libboost`
*   `python3` with `torch` and `matplotlib`

### 2. Compilation
```bash
cd externals/RePlAce
mkdir -p build && cd build
cmake ..
make -j8
```

### 3. Running ibm01 Benchmark
To run the ICCAD04 `ibm01` benchmark:
```bash
./replace -bmflag bookshelf \
  -aux /Users/roshaniruku/code/GraphPlace/data/ibm01_bookshelf/ibm01.aux \
  -den 1.0 -output ./output -onlyGP
```

### 4. Legalization & Scoring
We use a custom legalized designed for the **Macro Placement Challenge 2026** proxy cost metric.
```bash
python3 scripts/legalize_challenge.py \
  --pl externals/RePlAce/build/output/bookshelf/ibm01/experiment011/ibm01.eplace-gp.pl \
  --benchmark ibm01 \
  --output output/ibm01/ibm01_legalized.pt
```

## 📊 Evaluation
You can evaluate any placement using the competition harness:
```bash
cd externals/macro-place-challenge-2026
export PYTHONPATH=$PYTHONPATH:.
python3 -m macro_place.evaluate submissions/replace_legalized.py -b ibm01 --vis
```

## 📂 Repository Structure
*   `graphplace/`: Core GNN logic and graph converters.
*   `scripts/`: Legalization and data processing utilities.
*   `externals/`: Native engines (RePlAce, DREAMPlace, etc.) integrated directly into the source tree.
*   `data/`: Benchmark netlists and design constraints.
