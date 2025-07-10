#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define directories
CUDAQ_DIR="../../cuda-quantum"
QUAKE_DIR="../Dialect/Quake"
COMMON_DIR="../Dialect/Common"
CC_DIR="../Dialect/CC"

echo "Copying files from CUDA-Q source: $CUDAQ_DIR"

# Ensure source directories exist
if [ ! -d "$CUDAQ_DIR" ]; then
  echo "Error: CUDAQ_DIR '$CUDAQ_DIR' does not exist."
  exit 1
fi

# Copy to local QUAKE_DIR
echo "Copying Quake headers and TableGen files..."
cp "$CUDAQ_DIR/include/cudaq/Optimizer/Dialect/Quake/"*.h "$QUAKE_DIR"
cp "$CUDAQ_DIR/include/cudaq/Optimizer/Dialect/Quake/"*.td "$QUAKE_DIR"
cp "$CUDAQ_DIR/lib/Optimizer/Dialect/Quake/CanonicalPatterns.inc" "$QUAKE_DIR"
cp "$CUDAQ_DIR/lib/Optimizer/Dialect/Quake/"*.cpp "$QUAKE_DIR"
cp "$CUDAQ_DIR/lib/Optimizer/Builder/"*.cpp "$QUAKE_DIR"


echo "Copying other files..."
cp "$CUDAQ_DIR/include/cudaq/Optimizer/Builder/"*.h "$QUAKE_DIR"
cp "$CUDAQ_DIR/include/cudaq/Optimizer/CodeGen/"QIRFunctionNames.h "$QUAKE_DIR"
cp "$CUDAQ_DIR/include/cudaq/Optimizer/CodeGen/"QIROpaqueStructTypes.h "$QUAKE_DIR"

ls "$CUDAQ_DIR/include/cudaq/Optimizer/CodeGen/CudaqFunctionNames.h"
cp "$CUDAQ_DIR/include/cudaq/Optimizer/CodeGen/CudaqFunctionNames.h" "$QUAKE_DIR"
cp "$CUDAQ_DIR/include/cudaq/Support/SmallVector.h" "$QUAKE_DIR"

rm -f "$QUAKE_DIR/Marshal."*


# Copy to local CC_DIR
echo "Copying Quake source files..."

echo "Copying CC headers and TableGen files..."
cp "$CUDAQ_DIR/include/cudaq/Optimizer/Dialect/CC/"*.h "$CC_DIR"
cp "$CUDAQ_DIR/include/cudaq/Optimizer/Dialect/CC/"*.td "$CC_DIR"

echo "Copying CC source files..."
cp "$CUDAQ_DIR/lib/Optimizer/Dialect/CC/"*.cpp "$CC_DIR"


# Copy to Common
cp "$CUDAQ_DIR/include/cudaq/Optimizer/Dialect/Common/"*.h "$COMMON_DIR"
cp "$CUDAQ_DIR/include/cudaq/Optimizer/Dialect/Common/"*.td "$COMMON_DIR"


# Other 

bash modify_paths.h

echo " Transfer complete."
echo " Transformation of paths completed."




