# GraphPlace: A Heterogeneous GNN-RL Framework for VLSI Macro Placement

GraphPlace is a high-performance framework that couples a Heterogeneous Graph Neural Network (GNN) with a vectorized Reinforcement Learning pipeline to optimize VLSI macro placement. By treating macro placement as a refinement task in a continuous space, GraphPlace resolve overlaps and optimizes congestion on industrial-scale designs in minutes.

Macro placement is a critical stage in chip design where the positions of functional blocks (memory arrays, IP cores) determine the quality of downstream routing. While traditional solvers like RePlAce or Simulated Annealing are effective, they are "cold-start" optimizers that require hours of computation for every new design.

GraphPlace addresses these challenges by encoding the netlist as a heterogeneous graph and training a GNN-RL agent to learn transferable placement heuristics. Leveraging GPU-native HPWL calculations and C++ K-Nearest Neighbor (KNN) spatial queries, the framework achieves unprecedented scalability, handling designs with over 200,000 nodes (ibm18) on a single commercial GPU.

---

## Contents
1. [Architecture](#architecture)
   - [Introduction](#1-introduction)
   - [Graph Representation of Constraints](#2-graph-representation-of-constraints)
   - [PlaceGNN](#3-placegnn-the-core-intelligence)
   - [Reinforcement Learning Pipeline using GYM](#4-reinforcement-learning-pipeline)
2. [FAQ](#faq)
3. [Future Improvements and Compute](#future-improvements-and-compute)
4. [Installation and Setup](#installation-and-setup)
   - [Clone and Environment](#1-clone-and-environment)
   - [Running Training](#2-running-training-3600-epochs)
   - [Running Inference](#3-running-inference)
5. [Citations and Prior Work](#citations-and-prior-work)

---

## Architecture

### 1. Introduction
Traditional approaches to macro placement treat each new chip as a isolated optimization problem. GraphPlace frames this as a sequential decision-making problem. However, unlike prior RL placers that are bottlenecked by expensive O(N²) wirelength calculations, GraphPlace introduces a vectorized pipeline that allows the agent to reason about connectivity, geometric constraints, and spatial congestion in a fraction of the time.

### 2. Graph Representation of Constraints
<p align="center">
  <img src="https://github.com/user-attachments/assets/8319c9dc-d492-4e5d-a56c-fc08c8afad60" alt="GraphPlace Heterogeneous Topology" width="700">
</p>

The placement problem is encoded as a Heterogeneous Graph with three distinct node types: Macros, Nets, and Ports. This representation draws direct inspiration from the RL-MILP Solver (Lee & Kim, 2024), treating the macro placement canvas and netlist as a complex system of logical and integer constraints.

#### Star Expansion
GraphPlace utilizes a Star Expansion of the netlist hypergraph. In this bipartite-style topology, each net becomes a central hub connecting its participating members. However, unlike simpler models that only connect Macros to Nets, GraphPlace introduces Ports as an intermediate layer. This creates an effective Tripartite structure for message passing:
*   Macro <-> Net: captures global connectivity.
*   Port <-> Net: explicitly preserves physical pin offsets, which are critical for accurate wirelength and congestion estimation.

#### KNN Proximity Edges
To handle the physical reality of the chip canvas, GraphPlace superimposes a Spatial Layer on top of the netlist topology. During inference, the agent constructs Macro-to-Macro (near) edges using k-Nearest Neighbor (KNN) routines. This breaks the strict bipartite/tripartite abstraction of the netlist, allowing the GNN to reason about local density, spatial overlaps, and congestion zones in a non-linear continuous space.

By fusing these logical and spatial representations, the GNN learns to extract structural embeddings that guide the RL agent toward feasible, overlap-free solutions, paralleling the start primal heuristic methodology seen in modern MILP solvers.

### 3. PlaceGNN
The framework utilizes a custom heterogeneous GNN architecture (PlaceGNN). It acts as a structural encoder, using relation-specific message passing to preserve physical pin offsets while extracting logical connectivity. 
*   Structural Extraction: embeddings are processed by a Multi-Layer Perceptron (MLP) head.
*   Continuous Actions: The MLP outputs bounded "nudges" which are fine-grained displacements that allow the agent to slide macros away from overlap zones while minimizing the composite proxy cost.

### 4. Reinforcement Learning Pipeline using GYM
At each step, the RL agent evaluates the chip state and determines the optimal displacement for every macro. The environment provides rapid feedback using:
*   Half-Perimeter Wirelength (HPWL): GPU-vectorized for speed.
*   Overlap Penalties: Optimized for high-throughput training.
*   Zero-Shot Generalization: By training across the full IBM dataset, the agent learns generalized placement rules that can be applied to unseen topographies without retraining.

---

## FAQ

**Q: Why is GraphPlace better than traditional physical design solvers?**
**A:** Reusability. Traditional solvers start from a blank canvas every time. GraphPlace is a "warm-start" solver. It leverages pre-trained weights to output high-quality global placements in seconds, essentially acting as an AI-powered refinement layer for existing flow.

**Q: What sets GraphPlace apart from previous academic RL placers?**
**A:** GPU-Native Scalability. Most RL placers are bottlenecked by CPU-side math. We vectorized the HPWL calculation and offloaded spatial queries to compiled C++ KNN routines, allowing us to process massive benchmarks like ibm18 with over 200,000 nodes on a single NVIDIA L4 GPU.

---

## Future Improvements and Compute
While GraphPlace demonstrates state-of-the-art structural viability, its performance is currently bound by compute time. RL policies for VLSI typically require ~100,000 epochs for true mathematical convergence. Our current results (3,600 epochs on a single L4 over 4 hours) represent a baseline; we anticipate drastic improvements in placement quality and wirelength reduction when scaled to distributed A100/H100 clusters.

---

## Installation and Setup

### 1. Clone and Environment
```bash
git clone https://github.com/ForceDrift/GraphPlace.git
cd GraphPlace
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Running Training (3600+ Epochs)
```bash
python run_pipeline.py \
  --train-benchmarks ibm01 ibm02 ibm03 ibm04 ibm06 ibm07 ibm08 ibm09 ibm10 ibm11 ibm12 ibm13 ibm14 ibm15 ibm16 ibm17 ibm18 \
  --epochs 3600 \
  --steps 10
```

### 3. Running Inference
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/externals/macro-place-challenge-2026
python3 -m macro_place.evaluate submissions/gnn_placer_submission.py --all
```

---

## Citations and Prior Work
This research builds upon the foundations of deep reinforcement learning in physical design and heterogeneous graph representation learning:

*   AlphaChip: [A graph placement methodology for fast chip design](https://www.nature.com/articles/s41586-021-03544-w.epdf) (Nature, 2021)
*   Circuit Training: [Google Research Open-Source Infrastructure](https://github.com/google-research/circuit_training)
*   GNN-MILP Solver: [RL-MILP Solver: A Reinforcement Learning Approach for Solving Mixed-Integer Linear Programs with Graph Neural Networks](https://arxiv.org/pdf/2411.19517v4) (v4, 2024)

---
*Created by the GraphPlace Team for modern VLSI design automation.*
