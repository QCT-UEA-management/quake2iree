#!/bin/bash
# Rewrite upstream include paths inside a staging directory.
#
# Usage:
#   bash scripts/modify_paths.sh <staging-dir>
#
# Rules (applied in order — rule 1 creates the precondition for rule 4):
#   1. "cudaq/Optimizer/...  →  "...          (strips namespace prefix from all dialect headers)
#   2. "cudaq/Support/SmallVector.h"  →  "Dialect/Quake/SmallVector.h"
#   3. "cudaq/Frontend/nvqpp/AttributeNames.h"  →  "Dialect/Common/AttributeNames.h"
#   4. "Builder/Factory.h"  →  "Dialect/Quake/Factory.h"   (after rule 1 stripped "Optimizer/")

set -euo pipefail

STAGING="${1:?Usage: $0 <staging-dir>}"

if [ ! -d "$STAGING" ]; then
  echo "ERROR: staging directory not found: $STAGING"
  exit 1
fi

# Cross-platform sed
if [[ "$OSTYPE" == "darwin"* ]]; then
  SED_CMD=(sed -i '')
else
  SED_CMD=(sed -i)
fi

FILES=()
while IFS= read -r -d '' f; do
  FILES+=("$f")
done < <(find "$STAGING" -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.td" \) -print0)

if [ "${#FILES[@]}" -eq 0 ]; then
  echo "ERROR: no .cpp/.h/.td files found in $STAGING"
  exit 1
fi

echo "    Rewriting ${#FILES[@]} files in $STAGING"

# Rule 1: strip "cudaq/Optimizer/" prefix from all include paths
"${SED_CMD[@]}" 's|"cudaq/Optimizer/|"|g' "${FILES[@]}"

# Rule 2: SmallVector support header
"${SED_CMD[@]}" 's|"cudaq/Support/SmallVector\.h"|"Dialect/Quake/SmallVector.h"|g' "${FILES[@]}"

# Rule 3: AttributeNames (lives in Frontend/nvqpp/ upstream, maps to Dialect/Common/ here)
"${SED_CMD[@]}" 's|"cudaq/Frontend/nvqpp/AttributeNames\.h"|"Dialect/Common/AttributeNames.h"|g' "${FILES[@]}"

# Rule 4: Factory.h (after rule 1 stripped "Optimizer/", it now reads "Builder/Factory.h")
"${SED_CMD[@]}" 's|"Builder/Factory\.h"|"Dialect/Quake/Factory.h"|g' "${FILES[@]}"

echo "    Include paths rewritten."
