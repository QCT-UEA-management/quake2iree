#!/usr/bin/env python3
"""Controlled-Z gate.

Circuits tested:
  1. cz_on_11           - prepare |11>, CZ -> -|11>
  2. cz_on_superposition - prepare uniform superposition, CZ flips |11> phase

Run from the repo root:
  python examples/08_controlled_z.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cudaq
from examples.pipeline import (
    banner,
    run_statevector_kernel,
)


@cudaq.kernel
def cz_on_11():
    q = cudaq.qvector(2)
    x(q[0])
    x(q[1])
    cz(q[0], q[1])


@cudaq.kernel
def cz_on_superposition():
    q = cudaq.qvector(2)
    h(q[0])
    h(q[1])
    cz(q[0], q[1])


banner("Controlled-Z")
ok = True
ok = run_statevector_kernel(
    cz_on_11,
    "CZ |11> -> -|11>",
    expected={3: -1.0},
) and ok
ok = run_statevector_kernel(
    cz_on_superposition,
    "H,H,CZ |00> -> 0.5*(|00> + |01> + |10> - |11>)",
    expected={0: 0.5, 1: 0.5, 2: 0.5, 3: -0.5},
) and ok

banner("Done" if ok else "Some kernels failed - see above")
sys.exit(0 if ok else 1)
