#!/usr/bin/env python3
"""Reset qubits to |0>.

Circuits tested:
  1. reset_one  - prepare |1>, reset -> |0>
  2. reset_plus - prepare |+>, reset -> |0>

Run from the repo root:
  python examples/07_reset.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cudaq
from q2i import (
    banner,
    run_statevector_kernel,
)


@cudaq.kernel
def reset_one():
    q = cudaq.qvector(1)
    x(q[0])
    reset(q[0])


@cudaq.kernel
def reset_plus():
    q = cudaq.qvector(1)
    h(q[0])
    reset(q[0])


banner("Reset")
ok = True
ok = run_statevector_kernel(
    reset_one,
    "X then reset  |1> -> |0>",
    expected={0: 1.0},
) and ok
ok = run_statevector_kernel(
    reset_plus,
    "H then reset  |+> -> |0>",
    expected={0: 1.0},
) and ok

banner("Done" if ok else "Some kernels failed - see above")
sys.exit(0 if ok else 1)
