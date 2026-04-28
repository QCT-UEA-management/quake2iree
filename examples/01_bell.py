#!/usr/bin/env python3
"""Bell state — the simplest two-qubit entangled state.

Full pipeline:
  CUDA-Q kernel  ->  quake MLIR  ->  q2i-opt (quake-to-standard)
  ->  iree-compile  ->  iree-run-module

Run from the repo root:
  python examples/01_bell.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cudaq
from examples.pipeline import (
    banner,
    run_statevector_kernel,
)


# ---------------------------------------------------------------------------
# Kernel definition
# ---------------------------------------------------------------------------

@cudaq.kernel
def bell():
    q = cudaq.qvector(2)
    h(q[0])
    cx(q[0], q[1])


banner("Bell state")
ok = run_statevector_kernel(
    bell,
    "H + CNOT  |00> -> (|00> + |11>) / sqrt(2)",
    show_quake=True,
    show_lowered=True,
    print_raw_output=True,
)

banner("Done" if ok else "Some kernels failed - see above")
sys.exit(0 if ok else 1)
