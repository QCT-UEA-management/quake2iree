"""Benchmark runner: in-process compile, timing harness, and CLI entry point.

Compile a CUDA-Q kernel once to a VMFB, then call the compiled function
thousands of times without subprocess overhead. Results are collected as
TimingResult dataclasses that map directly to CSV rows.

Adding a new benchmark script
------------------------------
1. Write a factory function:  make_my_kernel(n_qubits: int) -> kernel
2. Call benchmark_cli() from __main__:

    if __name__ == "__main__":
        benchmark_cli("MyCircuit", make_my_kernel)

That is all. The CLI flags, backend resolution, CSV saving, and plotting are
handled here. To add a new hardware target, extend backends.py only — no
changes to this file are needed.
"""

from __future__ import annotations

import argparse
import csv
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

import numpy as np
import iree.compiler as irec
import iree.runtime as ireert

from q2i.lowering import emit_quake, entrypoint_name, q2i_convert, strip_cudaq_run_wrappers
from benchmarks.backends import (
    CUDAQ_CPU, CUDAQ_GPU, IREE_CPU, IREE_VMVX,
    CUDAQBackend, IREEBackend,
    KNOWN_BACKENDS, iree_cuda, resolve_backend,
)

_RESULTS_DIR = Path(__file__).parent / "results"


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class TimingResult:
    backend: str
    n_qubits: int
    compile_ms: float   # 0.0 for CUDA-Q (JIT cost not separately measurable)
    mean_us: float
    median_us: float
    std_us: float
    p95_us: float
    min_us: float
    n_samples: int


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------

def _lower_to_mlir(kernel) -> tuple[str, str]:
    """Lower a CUDA-Q kernel to standard MLIR. Returns (mlir_text, func_name)."""
    quake_ir = strip_cudaq_run_wrappers(emit_quake(kernel))
    func_name = entrypoint_name(quake_ir)
    if not func_name:
        raise RuntimeError("Could not find cudaq-entrypoint in quake IR")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        quake_file = tmp_path / "kernel.mlir"
        lowered_file = tmp_path / "kernel_lowered.mlir"
        quake_file.write_text(quake_ir)
        r = q2i_convert(quake_file, lowered_file)
        if r.returncode != 0:
            raise RuntimeError(f"q2i-opt failed:\n{r.stderr}")
        return lowered_file.read_text(), func_name


def compile_for_iree(kernel, backend: IREEBackend) -> tuple[bytes, str]:
    """Compile a CUDA-Q kernel to a VMFB for the given IREE backend.

    Returns (vmfb_bytes, function_name). Raises RuntimeError on failure.
    """
    mlir_text, func_name = _lower_to_mlir(kernel)
    vmfb = irec.compile_str(
        mlir_text,
        target_backends=[backend.target_backend],
        extra_args=list(backend.extra_compile_args),
    )
    return vmfb, func_name


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def time_iree(
    vmfb: bytes,
    func_name: str,
    backend: IREEBackend,
    n_qubits: int,
    *,
    warmup: int = 20,
    n: int = 200,
) -> TimingResult:
    """Time in-process IREE kernel execution and return statistics."""
    config = ireert.Config(driver_name=backend.driver)
    ctx = ireert.SystemContext(config=config)
    ctx.add_vm_module(ireert.VmModule.copy_buffer(ctx.instance, vmfb))
    fn = ctx.modules.module[func_name]

    # Build |00...0⟩ initial statevector: 1.0 at index 0, zeros elsewhere.
    # Passed as a runtime argument so IREE cannot constant-fold the circuit.
    n_f32 = 2 * (1 << n_qubits)
    init_sv = np.zeros(n_f32, dtype=np.float32)
    init_sv[0] = 1.0

    for _ in range(warmup):
        fn(init_sv)

    samples: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn(init_sv)
        samples.append((time.perf_counter() - t0) * 1e6)

    return _stats(backend.name, n_qubits, samples)


def time_cudaq(
    backend: CUDAQBackend,
    kernel,
    n_qubits: int,
    *,
    kernel_args: list | None = None,
    warmup: int = 20,
    n: int = 200,
) -> TimingResult:
    """Time cudaq.get_state() execution for a kernel and return statistics.

    kernel_args is forwarded as positional arguments to cudaq.get_state().
    """
    import cudaq

    cudaq.set_target(backend.target)
    args = kernel_args or []

    for _ in range(warmup):
        cudaq.get_state(kernel, *args)

    samples: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        cudaq.get_state(kernel, *args)
        samples.append((time.perf_counter() - t0) * 1e6)

    return _stats(backend.name, n_qubits, samples)


def _stats(backend: str, n_qubits: int, samples: list[float],
           compile_ms: float = 0.0) -> TimingResult:
    arr = sorted(samples)
    n = len(arr)
    mean = sum(arr) / n
    variance = sum((x - mean) ** 2 for x in arr) / n
    mid = n // 2
    median = arr[mid] if n % 2 else (arr[mid - 1] + arr[mid]) / 2.0
    p95 = arr[int(0.95 * (n - 1))]
    return TimingResult(
        backend=backend,
        n_qubits=n_qubits,
        compile_ms=compile_ms,
        mean_us=mean,
        median_us=median,
        std_us=variance ** 0.5,
        p95_us=p95,
        min_us=arr[0],
        n_samples=n,
    )


# ---------------------------------------------------------------------------
# Sweep — generic loop over qubit counts and backends
# ---------------------------------------------------------------------------

def run_sweep(
    kernel_factory: Callable[[int], object],
    qubit_counts: list[int],
    iree_backends: list[IREEBackend],
    cudaq_backends: list[CUDAQBackend],
    *,
    warmup: int,
    n_runs: int,
) -> list[TimingResult]:
    """Sweep qubit counts across backends, printing live results.

    kernel_factory(n_qubits) must return a compiled CUDA-Q kernel for n qubits.
    """
    results: list[TimingResult] = []

    for n in qubit_counts:
        print(f"\n  n_qubits = {n}")
        kernel = kernel_factory(n)

        for backend in iree_backends:
            try:
                t0 = time.perf_counter()
                vmfb, func_name = compile_for_iree(kernel, backend)
                compile_ms = (time.perf_counter() - t0) * 1e3
                r = time_iree(vmfb, func_name, backend, n, warmup=warmup, n=n_runs)
                r = replace(r, compile_ms=compile_ms)
                print(f"    {r.backend:<18} compile {r.compile_ms:6.0f} ms | "
                      f"median {r.median_us:8.1f}  p95 {r.p95_us:.1f} µs")
                results.append(r)
            except Exception as exc:
                print(f"    {backend.name:<18} SKIP  ({exc})")

        for backend in cudaq_backends:
            try:
                r = time_cudaq(backend, kernel, n, warmup=warmup, n=n_runs)
                print(f"    {r.backend:<18} median {r.median_us:8.1f}  p95 {r.p95_us:.1f} µs")
                results.append(r)
            except Exception as exc:
                print(f"    {backend.name:<18} SKIP  ({exc})")

    return results


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def save_csv(results: list[TimingResult], output_path: Path) -> None:
    """Write timing results to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "backend", "n_qubits", "compile_ms",
            "mean_us", "median_us", "std_us", "p95_us", "min_us", "n_samples",
        ])
        for r in results:
            writer.writerow([
                r.backend, r.n_qubits,
                f"{r.compile_ms:.1f}",
                f"{r.mean_us:.3f}", f"{r.median_us:.3f}", f"{r.std_us:.3f}",
                f"{r.p95_us:.3f}", f"{r.min_us:.3f}",
                r.n_samples,
            ])


# ---------------------------------------------------------------------------
# Backend resolution
# ---------------------------------------------------------------------------

def _resolve_backends(args) -> tuple[list[IREEBackend], list[CUDAQBackend]]:
    """Build backend lists from parsed CLI args.

    Looks up each requested name via resolve_backend() in backends.py.
    To add a new hardware target, extend backends.py only.
    """
    if args.backends is not None:
        requested = [s.strip() for s in args.backends.split(",")]
        unknown = set(requested) - KNOWN_BACKENDS
        if unknown:
            print(f"Warning: unknown backends ignored: {sorted(unknown)}")
        iree: list[IREEBackend] = []
        cudaq: list[CUDAQBackend] = []
        for name in requested:
            if name not in KNOWN_BACKENDS:
                continue
            b = resolve_backend(name)
            if isinstance(b, IREEBackend):
                iree.append(b)
            else:
                cudaq.append(b)
        return iree, cudaq

    # Default: CPU always; add CUDA unless suppressed
    iree = [IREE_CPU]
    cudaq = [CUDAQ_CPU]
    if not args.no_cuda:
        iree.append(iree_cuda())
        cudaq.append(CUDAQ_GPU)
    return iree, cudaq


# ---------------------------------------------------------------------------
# CLI entry point — shared by all benchmark scripts
# ---------------------------------------------------------------------------

def benchmark_cli(
    circuit_name: str,
    kernel_factory: Callable[[int], object],
) -> None:
    """Standard CLI entry point for a benchmark script.

    Handles argparse, backend resolution, the qubit-count sweep, CSV saving,
    and optional plotting. The caller only needs to supply a circuit name and
    a factory function that builds a CUDA-Q kernel for a given qubit count.

    Example usage in a benchmark script::

        if __name__ == "__main__":
            benchmark_cli("GHZ", make_ghz_kernel)

    The output CSV is written to benchmarks/results/<circuit>_YYYY-MM-DD.csv
    unless overridden with --output.
    """
    backends_list = ", ".join(sorted(KNOWN_BACKENDS))
    parser = argparse.ArgumentParser(
        description=f"{circuit_name} latency benchmark across IREE backends and CUDA-Q.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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
                            "comma-separated backends to run; overrides --no-cuda. "
                            f"Available: {backends_list}"
                        ))
    parser.add_argument("--plot", action="store_true",
                        help="generate a latency plot alongside the CSV")
    args = parser.parse_args()

    qubit_counts = [int(x) for x in args.qubit_counts.split(",")]
    iree_backends, cudaq_backends = _resolve_backends(args)

    print(f"Circuit:        {circuit_name}")
    print(f"Qubit counts:   {qubit_counts}")
    print(f"IREE backends:  {[b.name for b in iree_backends]}")
    print(f"CUDA-Q backends:{[b.name for b in cudaq_backends]}")
    print(f"Runs / warmup:  {args.runs} / {args.warmup}")

    results = run_sweep(
        kernel_factory, qubit_counts, iree_backends, cudaq_backends,
        warmup=args.warmup, n_runs=args.runs,
    )

    slug = circuit_name.lower().replace(" ", "_")
    output_path = (
        Path(args.output) if args.output
        else _RESULTS_DIR / f"{slug}_{date.today().isoformat()}.csv"
    )
    save_csv(results, output_path)
    print(f"\nResults: {output_path}")

    if args.plot:
        try:
            from benchmarks.plots import plot_latency, plot_compile_time
            plot_path = output_path.with_suffix(".png")
            plot_latency(output_path, plot_path)
            print(f"Plot:    {plot_path}")
            plot_compile_time(output_path)
        except ImportError as exc:
            print(f"Plotting skipped: {exc}")
