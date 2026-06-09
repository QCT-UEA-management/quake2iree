#!/usr/bin/env python3
"""GHZ state — three-qubit maximally entangled state.

Circuit:
    H(q[0])
    CNOT(ctrl=q[0], tgt=q[1])
    CNOT(ctrl=q[1], tgt=q[2])

Expected statevector: (|000⟩ + |111⟩) / √2

Full pipeline:
  CUDA-Q kernel  ->  quake MLIR  ->  q2i-opt (quake-to-standard)
  ->  iree-compile  ->  iree-run-module

Run from the repo root:
  python examples/02_ghz.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cudaq
from math import sqrt
from q2i import (
    banner,
    run_statevector_kernel,
)


# ---------------------------------------------------------------------------
# Kernel definition
# ---------------------------------------------------------------------------

@cudaq.kernel
def ghz():
    q = cudaq.qvector(3)
    h(q[0])
    cx(q[0], q[1])
    cx(q[1], q[2])


banner("GHZ state")
ok = run_statevector_kernel(
    ghz,
    "H + CNOT + CNOT  |000> -> (|000> + |111>) / sqrt(2)",
    show_quake=True,
    show_lowered=True,
    print_raw_output=True,
    expected={0: 1 / sqrt(2), 7: 1 / sqrt(2)},
)

banner("Done" if ok else "Some kernels failed - see above")
sys.exit(0 if ok else 1)
