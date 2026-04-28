#!/usr/bin/env python3
"""Sampling — draw measurement outcomes from a probability distribution.

The kernel is compiled and executed once by IREE, which returns the exact
statevector.  `shots` samples are then drawn from the implied probability
distribution in Python, producing a bitstring histogram — equivalent to
running the circuit `shots` times with wavefunction collapse.

Circuits tested:
  1. uniform_1q  — H |0⟩  → 50% |0⟩ / 50% |1⟩
  2. bell_sample — H + CNOT |00⟩  → 50% |00⟩ / 50% |11⟩
  3. ghz_sample  — H + 2×CNOT |000⟩  → 50% |000⟩ / 50% |111⟩

Run from the repo root:
  python examples/09_sampling.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cudaq
from examples.pipeline import banner, run_sample_kernel

SHOTS = 1000


@cudaq.kernel
def uniform_1q():
    q = cudaq.qvector(1)
    h(q[0])
    mz(q)


@cudaq.kernel
def bell_sample():
    q = cudaq.qvector(2)
    h(q[0])
    x.ctrl(q[0], q[1])
    mz(q)


@cudaq.kernel
def ghz_sample():
    q = cudaq.qvector(3)
    h(q[0])
    x.ctrl(q[0], q[1])
    x.ctrl(q[1], q[2])
    mz(q)


banner("Sampling from statevector")
ok = True
ok = run_sample_kernel(
    uniform_1q,
    "H |0⟩  →  50% |0⟩ / 50% |1⟩",
    SHOTS,
    expected_keys={"0", "1"},
) and ok
ok = run_sample_kernel(
    bell_sample,
    "Bell  →  50% |00⟩ / 50% |11⟩",
    SHOTS,
    expected_keys={"00", "11"},
) and ok
ok = run_sample_kernel(
    ghz_sample,
    "GHZ  →  50% |000⟩ / 50% |111⟩",
    SHOTS,
    expected_keys={"000", "111"},
) and ok

banner("Done" if ok else "Some kernels failed - see above")
sys.exit(0 if ok else 1)
