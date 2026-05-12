#!/usr/bin/env python3
"""Kernel execution latency vs qubit count.

Measures wall-clock time for a GHZ circuit across a range of qubit counts
and compares IREE backends against CUDA-Q native simulation. Compilation is
done once as setup; only the kernel call is timed.

Usage:
    python3 benchmarks/01_latency.py [options]

    --qubit-counts  Comma-separated list of qubit counts (default: 2,4,6,8,10,12)
    --runs          Timing iterations per measurement (default: 200)
    --warmup        Warmup iterations before timing (default: 20)
    --output        Output CSV path (default: benchmarks/results/latency_YYYY-MM-DD.csv)
    --no-cuda       Skip IREE CUDA and CUDA-Q GPU backends
    --plot          Generate a latency plot alongside the CSV
"""

import argparse
import csv
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cudaq

from benchmarks.backends import (
    CUDAQ_CPU,
    CUDAQ_GPU,
    IREE_CPU,
    CUDAQBackend,
    IREEBackend,
    iree_cuda,
)
from benchmarks.runner import TimingResult, compile_for_iree, time_cudaq, time_iree

RESULTS_DIR = Path(__file__).parent / "results"


# ---------------------------------------------------------------------------
# Circuit factory
# ---------------------------------------------------------------------------

def make_ghz_kernel(n_qubits: int):
    """Build a concrete GHZ kernel: H on q[0], then CNOT(q[0], q[i]) for i>0."""
    kernel = cudaq.make_kernel()
    q = kernel.qalloc(n_qubits)
    kernel.h(q[0])
    for i in range(1, n_qubits):
        kernel.cx(q[0], q[i])
    return kernel


# ---------------------------------------------------------------------------
# Benchmark loop
# ---------------------------------------------------------------------------

def run_benchmarks(
    qubit_counts: list[int],
    iree_backends: list[IREEBackend],
    cudaq_backends: list[CUDAQBackend],
    *,
    warmup: int,
    n_runs: int,
) -> list[TimingResult]:
    results: list[TimingResult] = []

    for n in qubit_counts:
        print(f"\n  n_qubits = {n}")
        kernel = make_ghz_kernel(n)

        for backend in iree_backends:
            try:
                vmfb, func_name = compile_for_iree(kernel, backend)
                r = time_iree(vmfb, func_name, backend, n, warmup=warmup, n=n_runs)
                print(f"    {r.backend:<18} {r.mean_us:8.1f} ± {r.std_us:.1f} µs")
                results.append(r)
            except Exception as exc:
                print(f"    {backend.name:<18} SKIP  ({exc})")

        for backend in cudaq_backends:
            try:
                r = time_cudaq(backend, kernel, n, warmup=warmup, n=n_runs)
                print(f"    {r.backend:<18} {r.mean_us:8.1f} ± {r.std_us:.1f} µs")
                results.append(r)
            except Exception as exc:
                print(f"    {backend.name:<18} SKIP  ({exc})")

    return results


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def save_csv(results: list[TimingResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["backend", "n_qubits", "mean_us", "std_us", "min_us", "n_samples"])
        for r in results:
            writer.writerow([
                r.backend, r.n_qubits,
                f"{r.mean_us:.3f}", f"{r.std_us:.3f}", f"{r.min_us:.3f}",
                r.n_samples,
            ])


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

_KNOWN_BACKENDS = {"iree-cpu", "iree-cuda", "iree-vmvx", "cudaq-cpu", "cudaq-gpu"}


def _resolve_backends(args) -> tuple[list[IREEBackend], list[CUDAQBackend]]:
    """Build backend lists from CLI args."""
    from benchmarks.backends import IREE_VMVX

    if args.backends is not None:
        requested = [s.strip() for s in args.backends.split(",")]
        unknown = set(requested) - _KNOWN_BACKENDS
        if unknown:
            print(f"Warning: unknown backends ignored: {sorted(unknown)}")
        iree: list[IREEBackend] = []
        cudaq: list[CUDAQBackend] = []
        for name in requested:
            if   name == "iree-cpu":   iree.append(IREE_CPU)
            elif name == "iree-cuda":  iree.append(iree_cuda())
            elif name == "iree-vmvx":  iree.append(IREE_VMVX)
            elif name == "cudaq-cpu":  cudaq.append(CUDAQ_CPU)
            elif name == "cudaq-gpu":  cudaq.append(CUDAQ_GPU)
        return iree, cudaq

    # Default: CPU always; add CUDA unless suppressed
    iree = [IREE_CPU]
    cudaq = [CUDAQ_CPU]
    if not args.no_cuda:
        iree.append(iree_cuda())
        cudaq.append(CUDAQ_GPU)
    return iree, cudaq


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--qubit-counts", default="2,4,6,8,10,12",
                        metavar="N,N,...", help="qubit counts to sweep")
    parser.add_argument("--runs", type=int, default=200,
                        metavar="N", help="timing iterations per measurement")
    parser.add_argument("--warmup", type=int, default=20,
                        metavar="N", help="warmup iterations before timing")
    parser.add_argument("--output", default=None,
                        metavar="PATH", help="output CSV path")
    parser.add_argument("--no-cuda", action="store_true",
                        help="skip CUDA backends (IREE and CUDA-Q)")
    parser.add_argument("--backends", default=None,
                        metavar="name,...",
                        help=(
                            "comma-separated list of backends to run; "
                            "overrides --no-cuda when present. "
                            "Available: iree-cpu, iree-cuda, iree-vmvx, "
                            "cudaq-cpu, cudaq-gpu"
                        ))
    parser.add_argument("--plot", action="store_true",
                        help="generate a latency plot alongside the CSV")
    args = parser.parse_args()

    qubit_counts = [int(x) for x in args.qubit_counts.split(",")]

    iree_backends, cudaq_backends = _resolve_backends(args)

    print(f"Circuit:        GHZ")
    print(f"Qubit counts:   {qubit_counts}")
    print(f"IREE backends:  {[b.name for b in iree_backends]}")
    print(f"CUDA-Q backends:{[b.name for b in cudaq_backends]}")
    print(f"Runs / warmup:  {args.runs} / {args.warmup}")

    results = run_benchmarks(
        qubit_counts, iree_backends, cudaq_backends,
        warmup=args.warmup, n_runs=args.runs,
    )

    output_path = (
        Path(args.output) if args.output
        else RESULTS_DIR / f"latency_{date.today().isoformat()}.csv"
    )
    save_csv(results, output_path)
    print(f"\nResults: {output_path}")

    if args.plot:
        try:
            from benchmarks.plots import plot_latency
            plot_path = output_path.with_suffix(".png")
            plot_latency(output_path, plot_path)
            print(f"Plot:    {plot_path}")
        except ImportError as exc:
            print(f"Plotting skipped: {exc}")


if __name__ == "__main__":
    main()
