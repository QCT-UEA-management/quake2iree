"""Benchmark runner: in-process compile and timing harness.

Compile a CUDA-Q kernel once to a VMFB, then call the compiled function
thousands of times without subprocess overhead. Results are collected as
TimingResult dataclasses that map directly to CSV rows.
"""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import iree.compiler as irec
import iree.runtime as ireert

from q2i.lowering import emit_quake, entrypoint_name, q2i_convert, strip_cudaq_run_wrappers
from benchmarks.backends import CUDAQBackend, IREEBackend


@dataclass
class TimingResult:
    backend: str
    n_qubits: int
    mean_us: float
    std_us: float
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

    for _ in range(warmup):
        fn()

    samples: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
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


def _stats(backend: str, n_qubits: int, samples: list[float]) -> TimingResult:
    mean = sum(samples) / len(samples)
    variance = sum((x - mean) ** 2 for x in samples) / len(samples)
    return TimingResult(
        backend=backend,
        n_qubits=n_qubits,
        mean_us=mean,
        std_us=variance ** 0.5,
        min_us=min(samples),
        n_samples=len(samples),
    )
