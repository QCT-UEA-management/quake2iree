#!/usr/bin/env python3
"""Swap gate.

Circuits tested:
  1. x_first_then_swap   - prepare |01>, swap q[0] and q[1] -> |10>
  2. superposition_swap  - prepare (|00> + |01>) / sqrt(2), swap -> (|00> + |10>) / sqrt(2)

These cases exercise a two-target gate without introducing measurement or
sampling semantics.

Run from the repo root:
  python examples/05_swap.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cudaq
from math import sqrt
from examples.pipeline import (
    banner,
    run_statevector_kernel,
)


@cudaq.kernel
def x_first_then_swap():
    q = cudaq.qvector(2)
    x(q[0])
    swap(q[0], q[1])


@cudaq.kernel
def superposition_swap():
    q = cudaq.qvector(2)
    h(q[0])
    swap(q[0], q[1])


banner("Swap gate")
ok = True
ok = run_statevector_kernel(
    x_first_then_swap,
    "X(q[0]) + SWAP(q[0], q[1])  |01> -> |10>",
    expected={2: 1.0},
) and ok
ok = run_statevector_kernel(
    superposition_swap,
    "H(q[0]) + SWAP(q[0], q[1])  (|00> + |01>) -> (|00> + |10>)",
    expected={0: 1 / sqrt(2), 2: 1 / sqrt(2)},
) and ok

banner("Done" if ok else "Some kernels failed - see above")
sys.exit(0 if ok else 1)
