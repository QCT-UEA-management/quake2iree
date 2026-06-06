# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

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
handled here. To add a new hardware target, extend q2i/backends.py only — no
changes to this file are needed.
"""

from __future__ import annotations

import argparse
import csv
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

import numpy as np
import iree.runtime as ireert

from q2i.backends import (
    CUDAQ_CPU, CUDAQ_GPU, IREE_CPU, IREE_VMVX,
    CUDAQBackend, IREEBackend,
    KNOWN_BACKENDS, iree_cuda, resolve_backend,
)
from q2i.compile import compile_mlir
from q2i.lowering import emit_quake, entrypoint_name, q2i_convert, strip_cudaq_run_wrappers

import tempfile

_RESULTS_DIR = Path(__file__).parent.parent / "benchmarks" / "results"


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
    p5_us: float
    p95_us: float
    min_us: float
    n_samples: int


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------

def _lower_to_mlir(kernel) -> tuple[str, str]:
    """Lower a CUDA-Q kernel to standard MLIR. Returns (mlir_text, func_name)."""
    quake_ir  = strip_cudaq_run_wrappers(emit_quake(kernel))
    func_name = entrypoint_name(quake_ir)
    if not func_name:
        raise RuntimeError("Could not find cudaq-entrypoint in quake IR")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path     = Path(tmp)
        quake_file   = tmp_path / "kernel.mlir"
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
    vmfb = compile_mlir(mlir_text, backend)
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
    kernel_args: list | None = None,
    warmup: int = 20,
    n: int = 200,
) -> TimingResult:
    """Time in-process IREE kernel execution and return statistics."""
    config = ireert.Config(driver_name=backend.driver)
    ctx    = ireert.SystemContext(config=config)
    ctx.add_vm_module(ireert.VmModule.copy_buffer(ctx.instance, vmfb))
    fn = ctx.modules.module[func_name]

    n_f32   = 2 * (1 << n_qubits)
    init_sv = np.zeros(n_f32, dtype=np.float32)
    init_sv[0] = 1.0

    args = list(kernel_args or []) + [init_sv]

    for _ in range(warmup):
        fn(*args)

    samples: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn(*args)
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
    """Time cudaq.get_state() execution for a kernel and return statistics."""
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
    n   = len(arr)
    mean     = sum(arr) / n
    variance = sum((x - mean) ** 2 for x in arr) / n
    mid      = n // 2
    median   = arr[mid] if n % 2 else (arr[mid - 1] + arr[mid]) / 2.0
    p5  = arr[int(0.05 * (n - 1))]
    p95 = arr[int(0.95 * (n - 1))]
    return TimingResult(
        backend=backend, n_qubits=n_qubits, compile_ms=compile_ms,
        mean_us=mean, median_us=median, std_us=variance ** 0.5,
        p5_us=p5, p95_us=p95, min_us=arr[0], n_samples=n,
    )


# ---------------------------------------------------------------------------
# Sweep
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
    """Sweep qubit counts across backends, printing live results."""
    results: list[TimingResult] = []

    for n in qubit_counts:
        print(f"\n  n_qubits = {n}")
        factory_result = kernel_factory(n)
        if isinstance(factory_result, tuple):
            kernel, kernel_args = factory_result
        else:
            kernel, kernel_args = factory_result, None

        for backend in iree_backends:
            try:
                t0 = time.perf_counter()
                vmfb, func_name = compile_for_iree(kernel, backend)
                compile_ms = (time.perf_counter() - t0) * 1e3
                r = time_iree(vmfb, func_name, backend, n,
                              kernel_args=kernel_args, warmup=warmup, n=n_runs)
                r = replace(r, compile_ms=compile_ms)
                print(f"    {r.backend:<18} compile {r.compile_ms:6.0f} ms | "
                      f"median {r.median_us:8.1f}  p95 {r.p95_us:.1f} µs")
                results.append(r)
            except Exception as exc:
                print(f"    {backend.name:<18} SKIP  ({exc})")

        for backend in cudaq_backends:
            try:
                r = time_cudaq(backend, kernel, n,
                               kernel_args=kernel_args, warmup=warmup, n=n_runs)
                print(f"    {r.backend:<18} median {r.median_us:8.1f}  p95 {r.p95_us:.1f} µs")
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
        writer.writerow([
            "backend", "n_qubits", "compile_ms",
            "mean_us", "median_us", "std_us", "p5_us", "p95_us", "min_us", "n_samples",
        ])
        for r in results:
            writer.writerow([
                r.backend, r.n_qubits,
                f"{r.compile_ms:.1f}",
                f"{r.mean_us:.3f}", f"{r.median_us:.3f}", f"{r.std_us:.3f}",
                f"{r.p5_us:.3f}", f"{r.p95_us:.3f}", f"{r.min_us:.3f}",
                r.n_samples,
            ])


# ---------------------------------------------------------------------------
# Backend resolution
# ---------------------------------------------------------------------------

def _resolve_backends(args) -> tuple[list[IREEBackend], list[CUDAQBackend]]:
    if args.backends is not None:
        requested = [s.strip() for s in args.backends.split(",")]
        unknown   = set(requested) - KNOWN_BACKENDS
        if unknown:
            print(f"Warning: unknown backends ignored: {sorted(unknown)}")
        iree:  list[IREEBackend]  = []
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

    iree  = [IREE_CPU]
    cudaq = [CUDAQ_CPU]
    if not args.no_cuda:
        iree.append(iree_cuda())
        cudaq.append(CUDAQ_GPU)
    return iree, cudaq


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def benchmark_cli(
    circuit_name: str,
    kernel_factory: Callable[[int], object],
    extra_plots: list[str] | None = None,
) -> None:
    """Standard CLI entry point for a benchmark script."""
    backends_list = ", ".join(sorted(KNOWN_BACKENDS))
    parser = argparse.ArgumentParser(
        description=f"{circuit_name} latency benchmark across IREE backends and CUDA-Q.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--qubit-counts", default="2,4,6,8,10,12",
                        metavar="N,N,...")
    parser.add_argument("--runs",   type=int, default=200, metavar="N")
    parser.add_argument("--warmup", type=int, default=20,  metavar="N")
    parser.add_argument("--output", default=None, metavar="PATH")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--backends", default=None, metavar="name,...",
                        help=f"Available: {backends_list}")
    parser.add_argument("--plot", action="store_true")
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
            from benchmarks.plots import plot_latency, plot_compile_time, plot_total_time
            plot_path = output_path.with_suffix(".png")
            plot_latency(output_path, plot_path)
            print(f"Plot:    {plot_path}")
            plot_compile_time(output_path)
            if extra_plots and "total_time" in extra_plots:
                plot_total_time(output_path)
        except ImportError as exc:
            print(f"Plotting skipped: {exc}")
