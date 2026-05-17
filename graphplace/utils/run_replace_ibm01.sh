#!/bin/bash

# Configuration
BENCHMARK="ibm01"
DATA_DIR="/Users/roshaniruku/code/GraphPlace/data/${BENCHMARK}_bookshelf"
OUTPUT_DIR="/Users/roshaniruku/code/GraphPlace/data/${BENCHMARK}_output"

mkdir -p "$OUTPUT_DIR"

# Step 1: Create the RePlAce configuration script (Tcl)
CAT_SCRIPT="$OUTPUT_DIR/replace.tcl"
cat <<EOF > "$CAT_SCRIPT"
# RePlAce Configuration for $BENCHMARK
set_output "$OUTPUT_DIR"
set_density 0.7
import_lef "/data/$BENCHMARK.nodes"
EOF

echo "Running RePlAce via Docker for $BENCHMARK..."

docker run --rm \
  -v "$DATA_DIR":/data \
  -v "$OUTPUT_DIR":/output \
  openroad/replace \
  /RePlAce/build/replace /data/$BENCHMARK.aux /output/ \
  -bmflag etc -density 0.7 -plot true

echo "Placement finished. Results are in $OUTPUT_DIR"
