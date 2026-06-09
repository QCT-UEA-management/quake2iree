# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

"""Shared plumbing for quake2iree examples.

Each helper wraps one pipeline stage and returns a CompletedProcess so the
caller can inspect returncode, stdout, and stderr without any policy baked in.
Display helpers (banner, show_ir, step_ok, step_fail) keep example scripts
readable without duplicating formatting logic.
"""

import re
import random
import shutil
import subprocess
import tempfile
import numpy as np
from pathlib import Path

from q2i.lowering import (  # noqa: F401 — re-exported for example scripts
    emit_quake,
    entrypoint_name,
    q2i_convert,
    strip_cudaq_run_wrappers,
)

IREE_BACKEND = "rocm" #"llvm-cpu"


def _nqubits_from_quake(quake_ir: str) -> int | None:
    """Extract the qubit count from the first quake.alloca in the IR."""
    m = re.search(r'quake\.alloca !quake\.veq<(\d+)>', quake_ir)
    return int(m.group(1)) if m else None


def _init_sv_input(n_qubits: int) -> str:
    """Build the iree-run-module --input value for |00...0⟩.

    Format: '<N>xf32=1 0 0 ... 0'  (2·2^n floats, first=1, rest=0).
    """
    n_f32 = 2 * (1 << n_qubits)
    return f"{n_f32}xf32=1 " + " ".join(["0"] * (n_f32 - 1))

def _init_sv_input(n_qubits: int, tmp_dir: Path) -> str:
    """Write the state vector to a .npy file and return the @path."""
    n_f32 = 2 * (1 << n_qubits)
    # Define the file path with .npy extension
    sv_file = tmp_dir / "sv_input.npy"
    # Create the array (using float32 as required by your tensor<...xf32>)
    data = np.zeros(n_f32, dtype=np.float32)
    data[0] = 1.0  # Set the |00...0> amplitude to 1.0
    # Save as a numpy file
    np.save(sv_file, data)
    return f"@{sv_file}"

def iree_compile(
    input_file: Path,
    output_file: Path,
    backend: str = IREE_BACKEND,
) -> subprocess.CompletedProcess:
    """Compile lowered MLIR to a VMFB with iree-compile."""
    tool = shutil.which("iree-compile")
    rocm_bc_path = "/opt/rocm-6.3.4/lib/llvm/lib/clang/18/lib/amdgcn/bitcode/"
    if not tool:
        raise RuntimeError("iree-compile not found in PATH")
    return subprocess.run(
        [tool, str(input_file), f"--iree-hal-target-backends={backend}", f"--iree-rocm-target=gfx90a", f"--iree-rocm-bc-dir={rocm_bc_path}", "-o", str(output_file)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )



def iree_run(
    vmfb_file: Path,
    function: str,
    args: list[str] | None = None,
) -> subprocess.CompletedProcess:
    """Execute a compiled IREE module.

    `args` is a list of --input values, e.g. ["3.14159::f64", "1.5707::f64"]
    for parametric kernels with runtime f64 arguments.
    """
    tool = shutil.which("iree-run-module")
    print(args)
    if not tool:
        raise RuntimeError("iree-run-module not found in PATH")
    cmd = [tool, f"--device=hip", f"--module={vmfb_file}", f"--function={function}"]
    if args:
        # Ensure that 'args' is actually a list of flags, not just the file path
        # It must be ['--input=@/tmp/...']
        for a in args:
            # If 'a' is already a full flag like '--input=@...', add as is
            # If 'a' is just the file path '@...', add '--input=' prefix
            if a.startswith("--input="):
                cmd.append(a)
            else:
                cmd.append(f"--input={a}")
    print(cmd)
    #if args:
    #    cmd += [f"--input={a}" for a in args]
    #for i in range(100):
    #    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

_WIDTH = 66


def banner(title: str):
    print(f"\n{'=' * _WIDTH}")
    print(f"  {title}")
    print(f"{'=' * _WIDTH}")


def show_ir(label: str, content: str, max_lines: int = 60):
    print(f"\n-- {label} --")
    lines = content.splitlines()
    print("\n".join(lines[:max_lines]))
    if len(lines) > max_lines:
        print(f"  ... ({len(lines) - max_lines} lines not shown)")


def step_ok(step: str, detail: str = ""):
    suffix = f" — {detail}" if detail else ""
    print(f"  PASS  {step}{suffix}")


def step_fail(step: str, stderr: str):
    print(f"  FAIL  {step}")
    if stderr.strip():
        for line in stderr.strip().splitlines():
            print(f"        {line}")


def parse_iree_f32_vector(iree_output: str) -> list[float]:
    """Extract the flat f32 values from iree-run-module output.

    iree-run-module prints lines like:
        result[0]: hal.buffer_view
        8xf32=0.707107 0 0 0 0 0 0.707107 0
    """
    for line in iree_output.splitlines():
        if "xf32=" in line:
            values_str = line.split("=", 1)[1]
            return [float(v) for v in values_str.split()]
    return []


def parse_iree_bool(iree_output: str) -> bool | None:
    """Extract a scalar boolean-like result from iree-run-module output."""
    m = re.search(r"i(?:1|32)=(true|false|0|1)", iree_output)
    if not m:
        return None
    value = m.group(1)
    return value in ("true", "1")


def print_statevector(iree_output: str, threshold: float = 1e-5):
    """Pretty-print a statevector from iree-run-module output.

    The statevector is stored as flat f32: [re0, im0, re1, im1, ...].
    Prints non-zero amplitudes in |k⟩ basis notation.
    """
    #values = parse_iree_f32_vector(iree_output)
    values = 0
    if not values or len(values) % 2 != 0:
        print("  (could not parse statevector)")
        return
    n_complex = len(values) // 2
    n_qubits = n_complex.bit_length() - 1
    print(f"  statevector ({n_qubits} qubits, {n_complex} amplitudes):")
    found_any = False
    for k in range(n_complex):
        re, im = values[2 * k], values[2 * k + 1]
        mag = (re ** 2 + im ** 2) ** 0.5
        if mag >= threshold:
            bits = format(k, f"0{n_qubits}b")
            if abs(im) < threshold:
                print(f"    |{bits}⟩  {re:+.6f}")
            else:
                print(f"    |{bits}⟩  {re:+.6f} {im:+.6f}i")
            found_any = True
    if not found_any:
        print("    (all amplitudes below threshold)")


def check_statevector(
    iree_output: str,
    expected: dict[int, complex],
    *,
    atol: float = 1e-4,
) -> bool:
    """Check final statevector amplitudes against a sparse expectation."""
    values = parse_iree_f32_vector(iree_output)
    if not values or len(values) % 2 != 0:
        step_fail("check statevector", "could not parse statevector")
        return False

    n_complex = len(values) // 2
    ok = True
    for k in range(n_complex):
        actual = complex(values[2 * k], values[2 * k + 1])
        want = expected.get(k, 0.0 + 0.0j)
        if abs(actual.real - want.real) > atol or abs(actual.imag - want.imag) > atol:
            bits = format(k, f"0{n_complex.bit_length() - 1}b")
            step_fail(
                "check statevector",
                f"|{bits}> expected {want.real:+.6f} {want.imag:+.6f}i, "
                f"got {actual.real:+.6f} {actual.imag:+.6f}i",
            )
            ok = False

    if ok:
        step_ok("check statevector")
    return ok


# ---------------------------------------------------------------------------
# Common example workflows
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
) -> bool:
    """Run one CUDA-Q kernel through q2i-opt, IREE compile, and IREE run.

    This helper is intentionally display-oriented: examples should stay useful
    as human-readable development probes. Later tests can reuse the lower-level
    stage runners above for stricter assertions.
    """
    banner(label)

    quake_ir = strip_cudaq_run_wrappers(emit_quake(kernel))
    func_name = entrypoint_name(quake_ir)
    if show_quake:
        show_ir("quake IR", quake_ir, max_lines=quake_max_lines)
    step_ok("emit_quake", f"entrypoint = {func_name}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        quake_file = tmp_path / "kernel.mlir"
        lowered_file = tmp_path / "kernel_lowered.mlir"
        vmfb_file = tmp_path / "kernel.vmfb"

        quake_file.write_text(quake_ir)

        result = q2i_convert(quake_file, lowered_file)
        if result.returncode != 0:
            step_fail("q2i-opt --quake-to-standard", result.stderr)
            print("\n  Pipeline stopped here - fix q2i-opt errors above to continue.")
            return False
        step_ok("q2i-opt --quake-to-standard")

        if show_lowered:
            lowered_ir = lowered_file.read_text()
            show_ir("lowered MLIR (standard dialects)", lowered_ir,
                    max_lines=lowered_max_lines)

        try:
            result = iree_compile(lowered_file, vmfb_file)
        except RuntimeError as exc:
            step_fail("iree-compile", str(exc))
            return False
        if result.returncode != 0:
            step_fail("iree-compile", result.stderr)
            print("\n  Pipeline stopped here - fix iree-compile errors above to continue.")
            return False
        step_ok("iree-compile", f"{vmfb_file.stat().st_size} bytes")

        # The sv arg is appended after existing kernel args, so order: kernel_args then sv.
        n_qubits = _nqubits_from_quake(quake_ir)
        n_qubits = _nqubits_from_quake(quake_ir)
        iree_args: list[str] = [f"{v}::f64" for v in kernel_args] if kernel_args else []
        
        if n_qubits is not None:
            # Pass the tmp_path so the function writes the file there
            sv_input_arg = _init_sv_input(n_qubits, tmp_path)
            iree_args.append(sv_input_arg)
            
        try:
            # iree_run logic remains the same; it just sees "--input=@/tmp/..."
            result = iree_run(vmfb_file, func_name, args=iree_args or None)
        #iree_args: list[str] = [f"{v}::f64" for v in kernel_args] if kernel_args else []
        #if n_qubits is not None:
        #    iree_args.append(_init_sv_input(n_qubits))
        #try:
        #    result = iree_run(vmfb_file, func_name, args=iree_args or None)
        except RuntimeError as exc:
            step_fail("iree-run-module", str(exc))
            return False
        if result.returncode != 0:
            step_fail("iree-run-module", result.stderr)
            return False
        step_ok("iree-run-module")

        if print_raw_output and result.stdout.strip():
            print(f"\n  raw output:\n  {result.stdout.strip()}")
            print()
        else:
            print()
        print_statevector(result.stdout)
        if expected is not None:
            return check_statevector(result.stdout, expected, atol=atol)

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
) -> bool:
    """Run one CUDA-Q kernel that returns a scalar i1 classical result."""
    banner(label)

    quake_ir = strip_cudaq_run_wrappers(emit_quake(kernel))
    func_name = entrypoint_name(quake_ir)
    if show_quake:
        show_ir("quake IR", quake_ir, max_lines=quake_max_lines)
    step_ok("emit_quake", f"entrypoint = {func_name}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        quake_file = tmp_path / "kernel.mlir"
        lowered_file = tmp_path / "kernel_lowered.mlir"
        vmfb_file = tmp_path / "kernel.vmfb"

        quake_file.write_text(quake_ir)

        result = q2i_convert(quake_file, lowered_file)
        if result.returncode != 0:
            step_fail("q2i-opt --quake-to-standard", result.stderr)
            print("\n  Pipeline stopped here - fix q2i-opt errors above to continue.")
            return False
        step_ok("q2i-opt --quake-to-standard")

        if show_lowered:
            lowered_ir = lowered_file.read_text()
            show_ir("lowered MLIR (standard dialects)", lowered_ir,
                    max_lines=lowered_max_lines)

        try:
            result = iree_compile(lowered_file, vmfb_file)
        except RuntimeError as exc:
            step_fail("iree-compile", str(exc))
            return False
        if result.returncode != 0:
            step_fail("iree-compile", result.stderr)
            print("\n  Pipeline stopped here - fix iree-compile errors above to continue.")
            return False
        step_ok("iree-compile", f"{vmfb_file.stat().st_size} bytes")

        n_qubits = _nqubits_from_quake(quake_ir)
        iree_args: list[str] = [f"{v}::f64" for v in kernel_args] if kernel_args else []
        if n_qubits is not None:
            iree_args.append(_init_sv_input(n_qubits))
        try:
            result = iree_run(vmfb_file, func_name, args=iree_args or None)
        except RuntimeError as exc:
            step_fail("iree-run-module", str(exc))
            return False
        if result.returncode != 0:
            step_fail("iree-run-module", result.stderr)
            return False
        step_ok("iree-run-module")

        print(f"\n  raw output:\n  {result.stdout.strip()}")
        actual = parse_iree_bool(result.stdout)
        if actual is None:
            step_fail("check i1 result", "could not parse i1 result")
            return False

        print(f"\n  classical result: {actual}")
        if expected is not None:
            if actual != expected:
                step_fail("check i1 result", f"expected {expected}, got {actual}")
                return False
            step_ok("check i1 result")

    return True


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def sample_from_statevector(
    iree_output: str,
    shots: int,
    *,
    seed: int | None = None,
) -> dict[str, int]:
    """Draw `shots` samples from the probability distribution of a statevector.

    The statevector is flat interleaved f32: [re0, im0, re1, im1, ...].
    p_k = re_k^2 + im_k^2.  Returns a bitstring → count histogram.
    """
    values = parse_iree_f32_vector(iree_output)
    if not values or len(values) % 2 != 0:
        return {}
    n_complex = len(values) // 2
    n_qubits = n_complex.bit_length() - 1
    probs = [values[2 * k] ** 2 + values[2 * k + 1] ** 2 for k in range(n_complex)]
    rng = random.Random(seed)
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
    """Check that only expected bitstrings appear and each count is statistically plausible.

    Uses a normal approximation: expected count ≈ shots / |expected_keys|,
    tolerance = sigma * sqrt(p * (1-p) * shots) where p = 1 / |expected_keys|.
    """
    unexpected = set(counts.keys()) - expected_keys
    if unexpected:
        step_fail("check samples", f"unexpected bitstrings: {sorted(unexpected)}")
        return False
    p = 1.0 / len(expected_keys)
    tol = sigma * (p * (1 - p) * shots) ** 0.5
    expected_count = p * shots
    ok = True
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


def run_sample_kernel(
    kernel,
    label: str,
    shots: int,
    *,
    show_quake: bool = True,
    quake_max_lines: int = 20,
    expected_keys: set[str] | None = None,
    seed: int | None = None,
) -> bool:
    """Run a CUDA-Q sampling kernel through the full pipeline and draw shots samples."""
    banner(label)

    quake_ir = strip_cudaq_run_wrappers(emit_quake(kernel))
    func_name = entrypoint_name(quake_ir)
    if show_quake:
        show_ir("quake IR", quake_ir, max_lines=quake_max_lines)
    step_ok("emit_quake", f"entrypoint = {func_name}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        quake_file = tmp_path / "kernel.mlir"
        lowered_file = tmp_path / "kernel_lowered.mlir"
        vmfb_file = tmp_path / "kernel.vmfb"

        quake_file.write_text(quake_ir)

        result = q2i_convert(quake_file, lowered_file)
        if result.returncode != 0:
            step_fail("q2i-opt --quake-to-standard", result.stderr)
            return False
        step_ok("q2i-opt --quake-to-standard")

        try:
            result = iree_compile(lowered_file, vmfb_file)
        except RuntimeError as exc:
            step_fail("iree-compile", str(exc))
            return False
        if result.returncode != 0:
            step_fail("iree-compile", result.stderr)
            return False
        step_ok("iree-compile", f"{vmfb_file.stat().st_size} bytes")

        n_qubits = _nqubits_from_quake(quake_ir)
        iree_args: list[str] = []
        if n_qubits is not None:
            iree_args.append(_init_sv_input(n_qubits))
        try:
            result = iree_run(vmfb_file, func_name, args=iree_args or None)
        except RuntimeError as exc:
            step_fail("iree-run-module", str(exc))
            return False
        if result.returncode != 0:
            step_fail("iree-run-module", result.stderr)
            return False
        step_ok("iree-run-module")

        counts = sample_from_statevector(result.stdout, shots, seed=seed)
        print(f"\n  samples ({shots} shots): {counts}")
        if expected_keys is not None:
            return check_sample_counts(counts, expected_keys, shots, sigma=5.0)

    return True
