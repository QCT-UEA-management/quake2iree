#!/usr/bin/env python3
"""Parametric ansatz latency sweep.

Measures kernel execution latency for a hardware-efficient ansatz (HEA)
with runtime rotation angles — the inner-loop pattern of VQE and QAOA.

Circuit structure per layer:
  Ry(θ_i) on every qubit, then CNOT(q_i, q_{i+1}) on adjacent pairs.

The circuit is compiled ONCE for IREE with the angles as runtime function
arguments (not baked-in constants).  Repeated calls reuse the compiled
kernel, demonstrating how IREE's AOT cost is amortised over many evaluations.

Use --plot to generate three figures:
  - execution latency (median ± p95 band) vs qubit count
  - AOT compilation time (IREE only) vs qubit count
  - total wall time (compile + N × exec) vs number of evaluations
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cudaq

from benchmarks.runner import benchmark_cli

_N_LAYERS = 2  # entangling layers; parameter count = n_qubits × _N_LAYERS


def make_ansatz_kernel(n_qubits: int):
    """Build an HEA kernel with runtime float parameters.

    Returns (kernel, fixed_angles).  The factory is called by run_sweep which
    detects the tuple and forwards fixed_angles to both time_iree and
    time_cudaq.  The specific angle values do not affect execution time.
    """
    n_params = n_qubits * _N_LAYERS
    result = cudaq.make_kernel(*([float] * n_params))
    kernel = result[0]
    thetas = result[1:]

    q = kernel.qalloc(n_qubits)

    param_idx = 0
    for _ in range(_N_LAYERS):
        for i in range(n_qubits):
            kernel.ry(thetas[param_idx], q[i])
            param_idx += 1
        for i in range(n_qubits - 1):
            kernel.cx(q[i], q[i + 1])

    # Fixed representative angles (π/4); any value gives the same exec time.
    fixed_angles = [0.785398] * n_params
    return kernel, fixed_angles


if __name__ == "__main__":
    benchmark_cli("Parametric-HEA", make_ansatz_kernel, extra_plots=["total_time"])
