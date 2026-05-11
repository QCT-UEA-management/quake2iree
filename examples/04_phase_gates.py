#!/usr/bin/env python3
"""Phase gates — S, T, and R1 with constant angles.

Circuits tested:
  1. h_then_s   — H then S on |0> -> (|0> + i|1>) / sqrt(2)
  2. h_then_t   — H then T on |0> -> (|0> + exp(i*pi/4)|1>) / sqrt(2)
  3. h_then_r1  — H then R1(pi/3) on |0> -> (|0> + exp(i*pi/3)|1>) / sqrt(2)

These are deterministic statevector checks for diagonal phase gates. They
extend the current single-qubit unitary path without introducing measurement
or sampling semantics yet.

Run from the repo root:
  python examples/04_phase_gates.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cudaq
from examples.pipeline import (
    banner,
    run_statevector_kernel,
)


@cudaq.kernel
def h_then_s():
    q = cudaq.qvector(1)
    h(q[0])
    s(q[0])


@cudaq.kernel
def h_then_t():
    q = cudaq.qvector(1)
    h(q[0])
    t(q[0])


@cudaq.kernel
def h_then_r1():
    q = cudaq.qvector(1)
    h(q[0])
    r1(math.pi / 3, q[0])


banner("Phase gates — S, T, R1")
ok = True
ok = run_statevector_kernel(
    h_then_s,
    "H+S        |0> -> (|0> + i|1>) / sqrt(2)",
    expected={0: 1 / math.sqrt(2), 1: 1j / math.sqrt(2)},
) and ok
ok = run_statevector_kernel(
    h_then_t,
    "H+T        |0> -> (|0> + exp(i*pi/4)|1>) / sqrt(2)",
    expected={0: 1 / math.sqrt(2), 1: 0.5 + 0.5j},
) and ok
ok = run_statevector_kernel(
    h_then_r1,
    "H+R1(pi/3) |0> -> (|0> + exp(i*pi/3)|1>) / sqrt(2)",
    expected={0: 1 / math.sqrt(2), 1: 0.5 / math.sqrt(2) + 0.5j * math.sqrt(3 / 2)},
) and ok

banner("Done" if ok else "Some kernels failed - see above")
sys.exit(0 if ok else 1)
