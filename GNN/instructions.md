# Quick run on ibm01
python3 GNN/placer_gnn.py --bench ibm01 --epochs 300

# Save model + placement
python3 GNN/placer_gnn.py --bench ariane133_ng45 --epochs 500 \
    --save GNN/model.pt --out-placement GNN/placement.pt \
    --device mps   # or cuda