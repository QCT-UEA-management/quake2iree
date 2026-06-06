# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

#!/usr/bin/env python3
"""Controlled-R1 gate — native CR1 support in quake2iree.

The CR1(λ) gate applies R1(λ) = [[1,0],[0,exp(iλ)]] to the target qubit
only when the control qubit is |1⟩.  When ctrl=|0⟩ the state is unchanged.

Circuits tested:
  1. cr1_on_11  — CR1(π/2) on |11⟩: ctrl=|1⟩ → i|11⟩  (phase applied)
  2. cr1_ctrl0  — CR1(π)   on |01⟩: ctrl=|0⟩ → |01⟩    (no-op check)
  3. cr1_super  — CR1(π/2) on (|01⟩+|11⟩)/√2 → (|01⟩+i|11⟩)/√2 (mixed)
  4. cr1_param  — parametric CR1(λ) at runtime, same as circuit 1 at λ=π/2

Qubit index convention (CUDA-Q default):
  state_index = q[0] + q[1]*2 + ...  (q[0] is least-significant bit)

Run from the repo root:
  python examples/11_cr1_gate.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cudaq
from q2i import banner, run_statevector_kernel

k = 1.0 / math.sqrt(2)


@cudaq.kernel
def cr1_on_11():
    """CR1(π/2) on |11⟩: ctrl=q[0]=1 triggers phase, result i|11⟩."""
    q = cudaq.qvector(2)
    x(q[0])
    x(q[1])
    r1.ctrl(math.pi / 2, q[0], q[1])


@cudaq.kernel
def cr1_ctrl0():
    """CR1(π) on |01⟩: ctrl=q[0]=0, gate is a no-op, result stays |01⟩."""
    q = cudaq.qvector(2)
    x(q[1])                        # only q[1]=1; q[0] stays |0⟩
    r1.ctrl(math.pi, q[0], q[1])


@cudaq.kernel
def cr1_super():
    """CR1(π/2) on (|01⟩+|11⟩)/√2: ctrl=q[0] in superposition.

    Preparation: H on q[0], X on q[1] → (|01⟩+|11⟩)/√2 in index notation.
    After CR1(π/2): |01⟩ unchanged (ctrl=0), |11⟩ → i|11⟩ (ctrl=1, tgt|1⟩).
    Result: (|01⟩ + i|11⟩)/√2.
    """
    q = cudaq.qvector(2)
    h(q[0])
    x(q[1])
    r1.ctrl(math.pi / 2, q[0], q[1])


@cudaq.kernel
def cr1_param(lam: float):
    """Parametric CR1(λ) on |11⟩: runtime angle, same physics as cr1_on_11."""
    q = cudaq.qvector(2)
    x(q[0])
    x(q[1])
    r1.ctrl(lam, q[0], q[1])


banner("Controlled-R1 (CR1) gate")
ok = True

# Circuit 1: |11⟩ → i|11⟩  (state index 3 = q[0]=1,q[1]=1)
ok = run_statevector_kernel(
    cr1_on_11,
    "CR1(π/2) on |11⟩ → i|11⟩",
    expected={3: 1j},
) and ok

# Circuit 2: ctrl=0 no-op, |01⟩ unchanged  (state 2 = q[0]=0, q[1]=1)
ok = run_statevector_kernel(
    cr1_ctrl0,
    "CR1(π)   on |01⟩, ctrl=0 → no-op, stays |01⟩",
    expected={2: 1.0},
) and ok

# Circuit 3: superposition  (state 2 = |01⟩, state 3 = |11⟩)
ok = run_statevector_kernel(
    cr1_super,
    "CR1(π/2) on (|01⟩+|11⟩)/√2 → (|01⟩ + i|11⟩)/√2",
    expected={2: k, 3: 1j * k},
) and ok

# Circuit 4: parametric (dynamic angle path)
ok = run_statevector_kernel(
    cr1_param,
    "CR1(λ=π/2) parametric on |11⟩ → i|11⟩",
    kernel_args=[math.pi / 2],
    expected={3: 1j},
) and ok

banner("Done" if ok else "Some kernels failed - see above")
sys.exit(0 if ok else 1)
