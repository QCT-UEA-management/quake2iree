#!/bin/bash
# Transfer the Quake and CC dialects from a cuda-quantum checkout.
#
# Strategy: pin SHA → atomic staging → build → pytest acceptance → git rollback on failure.
#
# Usage:
#   bash scripts/transfer_quake.sh
#   CUDAQ_DIR=/path/to/cuda-quantum bash scripts/transfer_quake.sh
#
# To upgrade cuda-quantum:
#   1. git -C "$CUDAQ_DIR" checkout <new-sha>
#   2. Update PINNED_SHA below.
#   3. Run this script and verify all tests pass.
#   4. Commit with: git add Dialect/ && git commit -m "vendor: update from cuda-quantum <sha>"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CUDAQ_DIR="${CUDAQ_DIR:-/workspaces/cuda-quantum}"

# ============================================================================
# Pinned commit — the last verified cuda-quantum version.
# Update this SHA only after a successful transfer + test run.
# ============================================================================
PINNED_SHA="501cca49d0012701615d695cd552b297705beb65"

# ---- Validate source repo --------------------------------------------------
if [ ! -d "$CUDAQ_DIR" ]; then
  echo "ERROR: cuda-quantum not found at $CUDAQ_DIR"
  echo "       Clone it there or set: CUDAQ_DIR=/path/to/cuda-quantum"
  exit 1
fi

ACTUAL_SHA=$(git -C "$CUDAQ_DIR" rev-parse HEAD)
if [ "$ACTUAL_SHA" != "$PINNED_SHA" ]; then
  echo "WARNING: cuda-quantum HEAD is $ACTUAL_SHA"
  echo "         pinned SHA is         $PINNED_SHA"
  echo ""
  echo "  To use pinned:  git -C $CUDAQ_DIR checkout $PINNED_SHA"
  echo "  To upgrade pin: edit PINNED_SHA in this script after tests pass."
  read -rp "Continue with $ACTUAL_SHA? [y/N] " reply
  [[ "$reply" =~ ^[Yy]$ ]] || exit 1
fi

echo "==> Source: cuda-quantum @ $ACTUAL_SHA"

# ---- Guard against uncommitted local changes in Dialect/ -------------------
if ! git -C "$PROJECT_DIR" diff --quiet HEAD -- Dialect/; then
  echo "WARNING: Dialect/ has uncommitted local changes."
  echo "         On failure the rollback (git checkout -- Dialect/) will discard them."
  read -rp "Continue? [y/N] " reply
  [[ "$reply" =~ ^[Yy]$ ]] || exit 1
fi

# ---- Atomic staging --------------------------------------------------------
STAGING=$(mktemp -d /tmp/q2i_staging_XXXXXX)
echo "==> Staging to $STAGING"

cleanup() {
  local code="$1"
  rm -rf "$STAGING"
  if [ "$code" != "0" ]; then
    echo ""
    echo "FAILED (exit $code) — restoring Dialect/ from last git commit."
    git -C "$PROJECT_DIR" checkout -- Dialect/
    echo "Dialect/ restored. No changes were kept."
  fi
}
trap 'cleanup $?' EXIT

mkdir -p "$STAGING/Quake" "$STAGING/CC" "$STAGING/Common"

# ---- Copy: what we need and nothing more -----------------------------------
echo "==> Copying Quake dialect"
# Headers and TableGen from include/
cp "$CUDAQ_DIR/include/cudaq/Optimizer/Dialect/Quake/"*.h  "$STAGING/Quake/"
cp "$CUDAQ_DIR/include/cudaq/Optimizer/Dialect/Quake/"*.td "$STAGING/Quake/"
# Implementation from lib/ (4 dialect .cpp files — Factory/Intrinsics are in Builder/, not here)
cp "$CUDAQ_DIR/lib/Optimizer/Dialect/Quake/"*.cpp           "$STAGING/Quake/"
cp "$CUDAQ_DIR/lib/Optimizer/Dialect/Quake/CanonicalPatterns.inc" "$STAGING/Quake/"
# Factory.h is included by QuakeOps.cpp; only the header is needed (not Factory.cpp)
cp "$CUDAQ_DIR/include/cudaq/Optimizer/Builder/Factory.h"   "$STAGING/Quake/"
# SmallVector.h wrapper (included by QuakeTypes.h and CCTypes.h)
cp "$CUDAQ_DIR/include/cudaq/Support/SmallVector.h"         "$STAGING/Quake/"

echo "==> Copying CC dialect"
cp "$CUDAQ_DIR/include/cudaq/Optimizer/Dialect/CC/"*.h      "$STAGING/CC/"
cp "$CUDAQ_DIR/include/cudaq/Optimizer/Dialect/CC/"*.td     "$STAGING/CC/"
cp "$CUDAQ_DIR/lib/Optimizer/Dialect/CC/"*.cpp              "$STAGING/CC/"

echo "==> Copying Common"
# InlinerInterface and Traits live in Dialect/Common/ upstream
cp "$CUDAQ_DIR/include/cudaq/Optimizer/Dialect/Common/InlinerInterface.h" "$STAGING/Common/"
cp "$CUDAQ_DIR/include/cudaq/Optimizer/Dialect/Common/Traits.h"           "$STAGING/Common/"
cp "$CUDAQ_DIR/include/cudaq/Optimizer/Dialect/Common/Traits.td"          "$STAGING/Common/"
# AttributeNames lives in Frontend/nvqpp/ upstream but is mapped to Dialect/Common/ here
cp "$CUDAQ_DIR/include/cudaq/Frontend/nvqpp/AttributeNames.h"             "$STAGING/Common/"

# ---- Rewrite include paths -------------------------------------------------
echo "==> Rewriting include paths"
bash "$SCRIPT_DIR/modify_paths.sh" "$STAGING"

# ---- Validate staged files -------------------------------------------------
echo "==> Validating staged files"
REQUIRED=(
  "Quake/QuakeOps.h"      "Quake/QuakeOps.cpp"
  "Quake/QuakeDialect.h"  "Quake/QuakeDialect.cpp"
  "Quake/QuakeTypes.h"    "Quake/QuakeTypes.cpp"
  "Quake/QuakeInterfaces.h" "Quake/QuakeInterfaces.cpp"
  "Quake/QuakeOps.td"     "Quake/QuakeTypes.td"
  "Quake/Factory.h"       "Quake/SmallVector.h"
  "Quake/CanonicalPatterns.inc"
  "CC/CCOps.h"    "CC/CCOps.cpp"
  "CC/CCDialect.h" "CC/CCDialect.cpp"
  "CC/CCTypes.h"  "CC/CCTypes.cpp"
  "Common/Traits.h" "Common/Traits.td"
  "Common/InlinerInterface.h" "Common/AttributeNames.h"
)
for f in "${REQUIRED[@]}"; do
  [ -f "$STAGING/$f" ] || { echo "ERROR: missing staged file: $f"; exit 1; }
done
echo "    All required files present."

# ---- Apply to Dialect/ -----------------------------------------------------
# CMakeLists.txt and QuakeCommon.cpp are not in staging — they are NOT overwritten.
echo "==> Applying to Dialect/"
cp "$STAGING/Quake/"* "$PROJECT_DIR/Dialect/Quake/"
cp "$STAGING/CC/"*    "$PROJECT_DIR/Dialect/CC/"
cp "$STAGING/Common/InlinerInterface.h" "$PROJECT_DIR/Dialect/Common/"
cp "$STAGING/Common/Traits.h"           "$PROJECT_DIR/Dialect/Common/"
cp "$STAGING/Common/Traits.td"          "$PROJECT_DIR/Dialect/Common/"
cp "$STAGING/Common/AttributeNames.h"   "$PROJECT_DIR/Dialect/Common/"

# ---- Build -----------------------------------------------------------------
echo "==> Building"
cd "$PROJECT_DIR"
bash build.sh

# ---- Acceptance tests ------------------------------------------------------
echo "==> Running acceptance tests"
python3 -m pytest test/test_circuits.py -q

# ---- Done ------------------------------------------------------------------
echo ""
echo "Transfer successful: cuda-quantum @ $ACTUAL_SHA"
echo ""
echo "  Review:  git diff Dialect/"
echo "  Commit:  git add Dialect/ && git commit -m 'vendor: update from cuda-quantum $ACTUAL_SHA'"

trap - EXIT
rm -rf "$STAGING"
