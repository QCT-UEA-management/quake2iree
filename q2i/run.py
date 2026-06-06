# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

"""In-process IREE execution and full-pipeline helpers.

Low-level API (compile once, run many times):
    amplitudes = run_statevector(vmfb, func_name, n_qubits, backend)
    bit        = run_i1(vmfb, func_name, n_qubits, backend)

High-level pipeline runners (lower → compile → run → check, for examples/tests):
    run_statevector_kernel(kernel, label, expected={...})
    run_i1_kernel(kernel, label, expected=bool)
    run_sample_kernel(kernel, label, shots, expected_keys={...})

Display helpers shared across examples:
    banner(title)
    show_ir(label, content)
    step_ok(step) / step_fail(step, detail)
"""

from __future__ import annotations

import random
import re
import tempfile
from pathlib import Path

import numpy as np
import iree.runtime as ireert

from q2i.backends import IREEBackend, IREE_CPU
from q2i.compile import compile_mlir
from q2i.lowering import (
    emit_quake,
    entrypoint_name,
    q2i_convert,
    strip_cudaq_run_wrappers,
)

# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

_WIDTH = 66


def banner(title: str) -> None:
    print(f"\n{'=' * _WIDTH}")
    print(f"  {title}")
    print(f"{'=' * _WIDTH}")


def show_ir(label: str, content: str, max_lines: int = 60) -> None:
    print(f"\n-- {label} --")
    lines = content.splitlines()
    print("\n".join(lines[:max_lines]))
    if len(lines) > max_lines:
        print(f"  ... ({len(lines) - max_lines} lines not shown)")


def step_ok(step: str, detail: str = "") -> None:
    suffix = f" — {detail}" if detail else ""
    print(f"  PASS  {step}{suffix}")


def step_fail(step: str, detail: str = "") -> None:
    print(f"  FAIL  {step}")
    if detail.strip():
        for line in detail.strip().splitlines():
            print(f"        {line}")


# ---------------------------------------------------------------------------
# Internal lowering helper
# ---------------------------------------------------------------------------

def _nqubits_from_quake(quake_ir: str) -> int | None:
    m = re.search(r'quake\.alloca !quake\.veq<(\d+)>', quake_ir)
    return int(m.group(1)) if m else None


def _full_lower(kernel) -> tuple[str, str, str]:
    """Lower a kernel to standard MLIR.

    Returns (quake_ir, func_name, mlir_text). Raises RuntimeError on failure.
    """
    quake_ir = strip_cudaq_run_wrappers(emit_quake(kernel))
    func_name = entrypoint_name(quake_ir)
    if not func_name:
        raise RuntimeError("Could not find cudaq-entrypoint in quake IR")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        quake_file  = tmp_path / "kernel.mlir"
        lowered_file = tmp_path / "kernel_lowered.mlir"
        quake_file.write_text(quake_ir)
        r = q2i_convert(quake_file, lowered_file)
        if r.returncode != 0:
            raise RuntimeError(r.stderr)
        return quake_ir, func_name, lowered_file.read_text()


# ---------------------------------------------------------------------------
# Low-level in-process runners
# ---------------------------------------------------------------------------

def _iree_ctx(vmfb: bytes, backend: IREEBackend) -> ireert.SystemContext:
    config = ireert.Config(driver_name=backend.driver)
    ctx = ireert.SystemContext(config=config)
    ctx.add_vm_module(ireert.VmModule.copy_buffer(ctx.instance, vmfb))
    return ctx


def _init_sv(n_qubits: int) -> np.ndarray:
    """Build the |00…0⟩ statevector as a flat f32 array [re0, im0, re1, im1, …]."""
    sv = np.zeros(2 * (1 << n_qubits), dtype=np.float32)
    sv[0] = 1.0
    return sv


def run_statevector(
    vmfb: bytes,
    func_name: str,
    n_qubits: int,
    backend: IREEBackend = IREE_CPU,
    *,
    kernel_args: list | None = None,
) -> np.ndarray:
    """Run a compiled statevector kernel; return amplitudes as complex64.

    Returns shape (2**n_qubits,) complex64.
    """
    ctx = _iree_ctx(vmfb, backend)
    fn  = ctx.modules.module[func_name]
    args = list(kernel_args or []) + [_init_sv(n_qubits)]
    flat = np.asarray(fn(*args)).astype(np.float32)
    return flat[0::2] + 1j * flat[1::2]


def run_i1(
    vmfb: bytes,
    func_name: str,
    n_qubits: int,
    backend: IREEBackend = IREE_CPU,
    *,
    kernel_args: list | None = None,
) -> bool:
    """Run a compiled measurement kernel; return the i1 result as bool."""
    ctx = _iree_ctx(vmfb, backend)
    fn  = ctx.modules.module[func_name]
    args = list(kernel_args or []) + [_init_sv(n_qubits)]
    return bool(fn(*args))


# ---------------------------------------------------------------------------
# Statevector display and checks
# ---------------------------------------------------------------------------

def print_statevector(amplitudes: np.ndarray, threshold: float = 1e-5) -> None:
    n_complex = len(amplitudes)
    n_qubits  = n_complex.bit_length() - 1
    print(f"  statevector ({n_qubits} qubits, {n_complex} amplitudes):")
    found_any = False
    for k, amp in enumerate(amplitudes):
        if abs(amp) >= threshold:
            bits = format(k, f"0{n_qubits}b")
            if abs(amp.imag) < threshold:
                print(f"    |{bits}⟩  {amp.real:+.6f}")
            else:
                print(f"    |{bits}⟩  {amp.real:+.6f} {amp.imag:+.6f}i")
            found_any = True
    if not found_any:
        print("    (all amplitudes below threshold)")


def check_statevector(
    amplitudes: np.ndarray,
    expected: dict[int, complex],
    *,
    atol: float = 1e-4,
) -> bool:
    n_complex = len(amplitudes)
    n_qubits  = n_complex.bit_length() - 1
    ok = True
    for k, amp in enumerate(amplitudes):
        want = complex(expected.get(k, 0.0))
        if abs(amp.real - want.real) > atol or abs(amp.imag - want.imag) > atol:
            bits = format(k, f"0{n_qubits}b")
            step_fail(
                "check statevector",
                f"|{bits}> expected {want.real:+.6f} {want.imag:+.6f}i, "
                f"got {amp.real:+.6f} {amp.imag:+.6f}i",
            )
            ok = False
    if ok:
        step_ok("check statevector")
    return ok


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def sample_from_statevector(
    amplitudes: np.ndarray,
    shots: int,
    *,
    seed: int | None = None,
) -> dict[str, int]:
    """Sample *shots* bitstrings from the probability distribution of *amplitudes*."""
    n_complex = len(amplitudes)
    n_qubits  = n_complex.bit_length() - 1
    probs = (amplitudes.real ** 2 + amplitudes.imag ** 2).tolist()
    rng   = random.Random(seed)
    counts: dict[str, int] = {}
    for k in rng.choices(range(n_complex), weights=probs, k=shots):
        bits = format(k, f"0{n_qubits}b")
        counts[bits] = counts.get(bits, 0) + 1
    return counts


def check_sample_counts(
    counts: dict[str, int],
    expected_keys: set[str],
    shots: int,
    *,
    sigma: float = 5.0,
) -> bool:
    unexpected = set(counts.keys()) - expected_keys
    if unexpected:
        step_fail("check samples", f"unexpected bitstrings: {sorted(unexpected)}")
        return False
    p   = 1.0 / len(expected_keys)
    tol = sigma * (p * (1 - p) * shots) ** 0.5
    expected_count = p * shots
    ok  = True
    for key in expected_keys:
        count = counts.get(key, 0)
        if abs(count - expected_count) > tol:
            step_fail(
                "check samples",
                f"|{key}⟩: got {count}, expected ≈{expected_count:.0f} ± {tol:.0f}",
            )
            ok = False
    if ok:
        step_ok("check samples", f"{shots} shots, {len(expected_keys)} outcomes")
    return ok


# ---------------------------------------------------------------------------
# High-level pipeline runners (examples + tests)
# ---------------------------------------------------------------------------

def run_statevector_kernel(
    kernel,
    label: str,
    *,
    kernel_args: list | None = None,
    show_quake: bool = True,
    show_lowered: bool = False,
    quake_max_lines: int = 20,
    lowered_max_lines: int = 60,
    print_raw_output: bool = False,
    expected: dict[int, complex] | None = None,
    atol: float = 1e-4,
    backend: IREEBackend = IREE_CPU,
) -> bool:
    """Run one kernel end-to-end and optionally check its statevector."""
    banner(label)

    try:
        quake_ir, func_name, mlir_text = _full_lower(kernel)
    except RuntimeError as exc:
        step_fail("q2i-opt --quake-to-standard", str(exc))
        print("\n  Pipeline stopped here - fix q2i-opt errors above to continue.")
        return False

    if show_quake:
        show_ir("quake IR", quake_ir, max_lines=quake_max_lines)
    step_ok("emit_quake", f"entrypoint = {func_name}")
    step_ok("q2i-opt --quake-to-standard")

    if show_lowered:
        show_ir("lowered MLIR (standard dialects)", mlir_text, max_lines=lowered_max_lines)

    try:
        vmfb = compile_mlir(mlir_text, backend)
    except Exception as exc:
        step_fail("iree-compile", str(exc))
        return False
    step_ok("iree-compile", f"{len(vmfb)} bytes")

    n_qubits = _nqubits_from_quake(quake_ir)
    if n_qubits is None:
        step_fail("run", "could not determine qubit count from quake IR")
        return False

    try:
        amplitudes = run_statevector(
            vmfb, func_name, n_qubits, backend, kernel_args=kernel_args
        )
    except Exception as exc:
        step_fail("iree-run-module", str(exc))
        return False
    step_ok("iree-run-module")

    if print_raw_output:
        flat = np.empty(len(amplitudes) * 2, dtype=np.float32)
        flat[0::2] = amplitudes.real
        flat[1::2] = amplitudes.imag
        label_str = f"{len(flat)}xf32=" + " ".join(f"{v}" for v in flat)
        print(f"\n  raw output:\n  EXEC @{func_name}")
        print(f"result[0]: hal.buffer_view\n{label_str}\n")

    print()
    print_statevector(amplitudes)
    if expected is not None:
        return check_statevector(amplitudes, expected, atol=atol)

    return True


def run_i1_kernel(
    kernel,
    label: str,
    *,
    kernel_args: list | None = None,
    show_quake: bool = True,
    show_lowered: bool = False,
    quake_max_lines: int = 24,
    lowered_max_lines: int = 60,
    expected: bool | None = None,
    backend: IREEBackend = IREE_CPU,
) -> bool:
    """Run a kernel that returns a scalar i1 classical result."""
    banner(label)

    try:
        quake_ir, func_name, mlir_text = _full_lower(kernel)
    except RuntimeError as exc:
        step_fail("q2i-opt --quake-to-standard", str(exc))
        print("\n  Pipeline stopped here - fix q2i-opt errors above to continue.")
        return False

    if show_quake:
        show_ir("quake IR", quake_ir, max_lines=quake_max_lines)
    step_ok("emit_quake", f"entrypoint = {func_name}")
    step_ok("q2i-opt --quake-to-standard")

    if show_lowered:
        show_ir("lowered MLIR (standard dialects)", mlir_text, max_lines=lowered_max_lines)

    try:
        vmfb = compile_mlir(mlir_text, backend)
    except Exception as exc:
        step_fail("iree-compile", str(exc))
        return False
    step_ok("iree-compile", f"{len(vmfb)} bytes")

    n_qubits = _nqubits_from_quake(quake_ir)
    if n_qubits is None:
        step_fail("run", "could not determine qubit count from quake IR")
        return False

    try:
        actual = run_i1(vmfb, func_name, n_qubits, backend, kernel_args=kernel_args)
    except Exception as exc:
        step_fail("iree-run-module", str(exc))
        return False
    step_ok("iree-run-module")

    print(f"\n  classical result: {actual}")
    if expected is not None:
        if actual != expected:
            step_fail("check i1 result", f"expected {expected}, got {actual}")
            return False
        step_ok("check i1 result")

    return True


def run_sample_kernel(
    kernel,
    label: str,
    shots: int,
    *,
    show_quake: bool = True,
    quake_max_lines: int = 20,
    expected_keys: set[str] | None = None,
    seed: int | None = None,
    backend: IREEBackend = IREE_CPU,
) -> bool:
    """Run a sampling kernel and draw *shots* samples from its statevector."""
    banner(label)

    try:
        quake_ir, func_name, mlir_text = _full_lower(kernel)
    except RuntimeError as exc:
        step_fail("q2i-opt --quake-to-standard", str(exc))
        return False

    if show_quake:
        show_ir("quake IR", quake_ir, max_lines=quake_max_lines)
    step_ok("emit_quake", f"entrypoint = {func_name}")
    step_ok("q2i-opt --quake-to-standard")

    try:
        vmfb = compile_mlir(mlir_text, backend)
    except Exception as exc:
        step_fail("iree-compile", str(exc))
        return False
    step_ok("iree-compile", f"{len(vmfb)} bytes")

    n_qubits = _nqubits_from_quake(quake_ir)
    if n_qubits is None:
        step_fail("run", "could not determine qubit count from quake IR")
        return False

    try:
        amplitudes = run_statevector(vmfb, func_name, n_qubits, backend)
    except Exception as exc:
        step_fail("iree-run-module", str(exc))
        return False
    step_ok("iree-run-module")

    counts = sample_from_statevector(amplitudes, shots, seed=seed)
    print(f"\n  samples ({shots} shots): {counts}")
    if expected_keys is not None:
        return check_sample_counts(counts, expected_keys, shots, sigma=5.0)

    return True
