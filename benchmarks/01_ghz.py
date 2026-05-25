#!/usr/bin/env python3
"""GHZ latency sweep across qubit counts and backends.

Measures wall-clock time for a GHZ circuit (H on q[0], then CNOT(q[0], q[i])
for each subsequent qubit) across a range of qubit counts. Compilation is done
once as setup; only the kernel call is timed.

Usage:
    python3 benchmarks/01_ghz.py [options]
    python3 benchmarks/01_ghz.py --help
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cudaq

from benchmarks.runner import benchmark_cli


def make_ghz_kernel(n_qubits: int):
    """Build a GHZ kernel: H on q[0], then CNOT(q[0], q[i]) for i > 0."""
    kernel = cudaq.make_kernel()
    q = kernel.qalloc(n_qubits)
    kernel.h(q[0])
    for i in range(1, n_qubits):
        kernel.cx(q[0], q[i])
    return kernel


if __name__ == "__main__":
    benchmark_cli("GHZ", make_ghz_kernel)
