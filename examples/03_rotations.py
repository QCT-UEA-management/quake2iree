#!/usr/bin/env python3
"""Rotation gates — RX, RY, RZ with constant angles.

Circuits tested:
  1. rx_half_turn  — RX(π) on |0⟩ → |1⟩  (should flip qubit: amplitude at |1⟩)
  2. ry_quarter    — RY(π/2) on |0⟩ → (|0⟩ + |1⟩)/√2  (like H but real entries)
  3. rz_phase      — H then RZ(π/2) on |0⟩ → (|0⟩ + i|1⟩)/√2  (adds π/2 phase)

These cover all three Bloch-sphere rotation axes and verify that constant
angles are extracted and matrix elements computed correctly at lowering time.

Run from the repo root:
  python examples/03_rotations.py
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


# ---------------------------------------------------------------------------
# 1. RX(π) — full bit-flip around X axis: |0⟩ → -i|1⟩
#    (global phase aside, amplitude lands at index 1)
# ---------------------------------------------------------------------------

@cudaq.kernel
def rx_half_turn():
    q = cudaq.qvector(1)
    rx(math.pi, q[0])   # RX(π) = -i·X


# ---------------------------------------------------------------------------
# 2. RY(π/2) — quarter turn around Y axis: |0⟩ → (|0⟩ + |1⟩)/√2
#    Like Hadamard but with purely real matrix entries.
# ---------------------------------------------------------------------------

@cudaq.kernel
def ry_quarter():
    q = cudaq.qvector(1)
    ry(math.pi / 2, q[0])   # RY(π/2)


# ---------------------------------------------------------------------------
# 3. H then RZ(π/2) — starts in |+⟩, adds a relative phase:
#    |0⟩ → H → (|0⟩+|1⟩)/√2 → RZ(π/2) → (e^{-iπ/4}|0⟩ + e^{iπ/4}|1⟩)/√2
#    In terms of amplitudes: re₀≈0.5, im₀≈-0.5, re₁≈0.5, im₁≈0.5
# ---------------------------------------------------------------------------

@cudaq.kernel
def h_then_rz():
    q = cudaq.qvector(1)
    h(q[0])
    rz(math.pi / 2, q[0])   # RZ(π/2)


banner("Rotation gates — RX, RY, RZ")
ok = True
ok = run_statevector_kernel(rx_half_turn,  "RX(π)      |0⟩ → -i|1⟩") and ok
ok = run_statevector_kernel(ry_quarter,    "RY(π/2)    |0⟩ → (|0⟩+|1⟩)/√2") and ok
ok = run_statevector_kernel(h_then_rz,     "H+RZ(π/2)  |0⟩ → (e^{-iπ/4}|0⟩ + e^{iπ/4}|1⟩)/√2") and ok

banner("Done" if ok else "Some kernels failed - see above")
sys.exit(0 if ok else 1)
