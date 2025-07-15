#!/bin/bash

# Path to the compiled tool
Q2I_OPT=../build/tool/q2i-opt

# Input file (first argument) and optional output file (second argument)
INPUT_FILE=${1:-test.mlir}
OUTPUT_FILE=${2:-output.mlir}

# Check that the binary exists
if [ ! -x "$Q2I_OPT" ]; then
  echo "Error: $Q2I_OPT not found or not executable."
  exit 1
fi

# Run the tool with the Quake-to-Standard pass
"$Q2I_OPT" "$INPUT_FILE" --quake-to-standard -o "$OUTPUT_FILE"

# Report success
echo "✅ Transformed MLIR written to $OUTPUT_FILE"
