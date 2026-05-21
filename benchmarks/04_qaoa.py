# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

#!/usr/bin/env python3
"""QAOA (Quantum Approximate Optimization Algorithm) latency sweep.

Benchmarks the QAOA-MaxCut circuit for a ring (cycle) graph with p=1 layers.
This script measures single-circuit execution latency: the kernel is compiled
once and then called repeatedly with fixed representative angles to collect
stable timing statistics.  No optimizer loop is run here.

The circuit is representative of the inner loop of a real QAOA variational
sweep, where the same compiled circuit would be evaluated at many different
angle values.  The compile_ms column in the CSV captures the one-time AOT
cost; the median_us column captures the per-call execution cost.

Circuit for one layer on an n-qubit ring graph:
  Preparation:  H(q[i])  for all i
  Cost unitary: for each edge (i, i+1 mod n):
                  CNOT(q[i], q[i+1]); Rz(γ, q[i+1]); CNOT(q[i], q[i+1])
  Mixer:         Rx(β, q[i]) for all i

Parameters: [γ, β] passed as runtime f64 arguments (2 per layer).
Gate count:  n H + 2·n CNOT + n Rz + n Rx  →  O(n) gates (linear in qubits).

The ring graph is chosen because it gives exactly n edges for n nodes, so gate
count scales cleanly.  The edge (n-1, 0) wraps around, matching the periodic
boundary convention common in condensed-matter-inspired benchmarks.

The --plot flag generates three figures (latency, compile time, total wall time)
because the total-time crossover chart is the core narrative for this circuit.

Run from the repo root:
  python3 benchmarks/04_qaoa.py [options]
  python3 benchmarks/04_qaoa.py --help
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cudaq

from benchmarks.runner import benchmark_cli

_P_LAYERS = 1  # QAOA depth; parameter count = 2 * _P_LAYERS


def make_qaoa_kernel(n_qubits: int):
    """Build a QAOA-MaxCut kernel for a ring graph with _P_LAYERS layers.

    Returns (kernel, fixed_angles).  fixed_angles are representative values
    (γ ≈ π/8, β ≈ π/4) — execution time is angle-independent for this circuit.
    """
    n_params = 2 * _P_LAYERS  # [γ_0, ..., γ_{p-1}, β_0, ..., β_{p-1}]
    result = cudaq.make_kernel(*([float] * n_params))
    kernel = result[0]
    params = result[1:]

    q = kernel.qalloc(n_qubits)

    # Initial state: uniform superposition |+>^n
    for i in range(n_qubits):
        kernel.h(q[i])

    # p QAOA layers
    edges = [(i, (i + 1) % n_qubits) for i in range(n_qubits)]
    for layer in range(_P_LAYERS):
        gamma = params[layer]
        beta  = params[_P_LAYERS + layer]

        # Cost unitary U_C(γ): ZZ on each ring edge via CNOT–Rz–CNOT
        for (i, j) in edges:
            kernel.cx(q[i], q[j])
            kernel.rz(gamma, q[j])
            kernel.cx(q[i], q[j])

        # Mixer unitary U_B(β): X-rotation on every qubit
        for i in range(n_qubits):
            kernel.rx(beta, q[i])

    # Representative angles (π/8, π/4) — standard QAOA warm-start values
    fixed_angles = [math.pi / 8] * _P_LAYERS + [math.pi / 4] * _P_LAYERS
    return kernel, fixed_angles


if __name__ == "__main__":
    benchmark_cli("QAOA-MaxCut", make_qaoa_kernel, extra_plots=["total_time"])
