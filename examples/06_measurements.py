#!/usr/bin/env python3
"""Z-basis measurement returning classical bits.

Circuits tested:
  1. measure_zero - measure |0> -> False
  2. measure_one  - prepare |1>, measure -> True
  3. measure_plus - prepare |+>, deterministic tie -> False

This first measurement example is intentionally deterministic. It only checks
basis states plus one explicit superposition tie case, so no random sampling
support is needed yet.

Run from the repo root:
  python examples/06_measurements.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cudaq
from examples.pipeline import (
    banner,
    run_i1_kernel,
)


@cudaq.kernel
def measure_zero() -> bool:
    q = cudaq.qvector(1)
    return mz(q[0])


@cudaq.kernel
def measure_one() -> bool:
    q = cudaq.qvector(1)
    x(q[0])
    return mz(q[0])


@cudaq.kernel
def measure_plus() -> bool:
    q = cudaq.qvector(1)
    h(q[0])
    return mz(q[0])


banner("Z-basis measurements")
ok = True
ok = run_i1_kernel(
    measure_zero,
    "MZ |0> -> False",
    expected=False,
) and ok
ok = run_i1_kernel(
    measure_one,
    "X then MZ |0> -> True",
    expected=True,
) and ok
ok = run_i1_kernel(
    measure_plus,
    "H then MZ |0> -> False  (deterministic 50/50 tie)",
    expected=False,
) and ok

banner("Done" if ok else "Some kernels failed - see above")
sys.exit(0 if ok else 1)
