#!/bin/bash

# Configuration
BENCHMARK="ibm01"
DATA_DIR="/Users/roshaniruku/code/GraphPlace/data/${BENCHMARK}_bookshelf"
OUTPUT_DIR="/Users/roshaniruku/code/GraphPlace/data/${BENCHMARK}_output"

mkdir -p "$OUTPUT_DIR"

# Step 1: Create the RePlAce configuration script (Tcl)
# RePlAce standalone uses a Tcl script to define the run.
CAT_SCRIPT="$OUTPUT_DIR/replace.tcl"
cat <<EOF > "$CAT_SCRIPT"
# RePlAce Configuration for $BENCHMARK
set_output "$OUTPUT_DIR"
set_density 0.7
# Bookshelf files are in /data
import_lef "/data/$BENCHMARK.nodes"
# Actually, RePlAce standalone reading Bookshelf uses a different flow
# or it can take the .aux file directly.
EOF

# Actually, the 'replace' binary usually takes the .aux file and arguments.
# Let's check how the binary is usually called.
# Based on RePlAce README:
# ./replace <aux_file> <output_dir> [options]

# We will run it via Docker. 
# We mount the data directory to /data and run the binary.

echo "Running RePlAce via Docker for $BENCHMARK..."

docker run --rm \
  -v "$DATA_DIR":/data \
  -v "$OUTPUT_DIR":/output \
  openroad/replace \
  /RePlAce/build/replace /data/$BENCHMARK.aux /output/ \
  -bmflag etc -density 0.7 -plot true

echo "Placement finished. Results are in $OUTPUT_DIR"
