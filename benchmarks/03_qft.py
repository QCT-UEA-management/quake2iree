#!/usr/bin/env python3
"""QFT (Quantum Fourier Transform) latency sweep.

Measures kernel execution latency for the standard QFT circuit across a range
of qubit counts.  The QFT is a cornerstone of quantum algorithms (Shor's, QPE,
HHL) and is the most demanding fixed-gate benchmark in this suite:

  Gate count: n Hadamards + n(n-1)/2 CR1 gates  →  O(n²) gates total.
  CR1 angles: 2π/2^(m-k+1) for each (k, m) pair — all compile-time constants.

Circuit structure for k = 0, 1, ..., n-1:
  H(q[k])
  CR1(2π/4,   q[k+1], q[k])
  CR1(2π/8,   q[k+2], q[k])
  ...
  CR1(2π/2^(n-k), q[n-1], q[k])

Note: bit-reversal swap layer is omitted — it contributes only O(n) swaps and
does not affect the execution-time scaling comparison between backends.

All CR1 angles fold to compile-time f32 constants in the IREE lowering pass,
so IREE never emits runtime trig calls for this circuit.

Run from the repo root:
  python3 benchmarks/03_qft.py [options]
  python3 benchmarks/03_qft.py --help
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cudaq

from q2i import benchmark_cli


def make_qft_kernel(n_qubits: int):
    """Build a flat QFT kernel with all angles as compile-time constants.

    Uses cudaq.make_kernel() so the entire circuit is a single function with
    no sub-kernel calls.  This is required by the quake2iree lowering pass.
    """
    kernel = cudaq.make_kernel()
    q = kernel.qalloc(n_qubits)

    for k in range(n_qubits):
        kernel.h(q[k])
        for m in range(k + 1, n_qubits):
            angle = 2.0 * math.pi / (2 ** (m - k + 1))
            kernel.cr1(angle, q[m], q[k])  # ctrl=q[m], tgt=q[k]

    return kernel


if __name__ == "__main__":
    benchmark_cli("QFT", make_qft_kernel)
