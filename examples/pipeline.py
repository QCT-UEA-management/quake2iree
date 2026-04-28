"""Shared plumbing for quake2iree examples.

Each helper wraps one pipeline stage and returns a CompletedProcess so the
caller can inspect returncode, stdout, and stderr without any policy baked in.
Display helpers (banner, show_ir, step_ok, step_fail) keep example scripts
readable without duplicating formatting logic.
"""

import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent
Q2I_OPT = ROOT / "build" / "tool" / "q2i-opt"
IREE_BACKEND = "llvm-cpu"


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def emit_quake(kernel) -> str:
    """Compile a CUDA-Q kernel and return its raw quake MLIR string."""
    kernel.compile()
    return str(kernel.qkeModule)


def entrypoint_name(quake_ir: str) -> str | None:
    """Return the symbol name of the cudaq-entrypoint function, or None."""
    m = re.search(r'func\.func @(\S+?)\(.*?"cudaq-entrypoint"', quake_ir)
    return m.group(1) if m else None


def q2i_convert(input_file: Path, output_file: Path) -> subprocess.CompletedProcess:
    """Lower quake dialect to standard dialects via q2i-opt."""
    return subprocess.run(
        [str(Q2I_OPT), str(input_file), "--quake-to-standard", "-o", str(output_file)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def iree_compile(
    input_file: Path,
    output_file: Path,
    backend: str = IREE_BACKEND,
) -> subprocess.CompletedProcess:
    """Compile lowered MLIR to a VMFB with iree-compile."""
    tool = shutil.which("iree-compile")
    if not tool:
        raise RuntimeError("iree-compile not found in PATH")
    return subprocess.run(
        [tool, str(input_file), f"--iree-hal-target-backends={backend}", "-o", str(output_file)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def iree_run(vmfb_file: Path, function: str) -> subprocess.CompletedProcess:
    """Execute a compiled IREE module."""
    tool = shutil.which("iree-run-module")
    if not tool:
        raise RuntimeError("iree-run-module not found in PATH")
    return subprocess.run(
        [tool, f"--module={vmfb_file}", f"--function={function}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


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


def print_statevector(iree_output: str, threshold: float = 1e-5):
    """Pretty-print a statevector from iree-run-module output.

    The statevector is stored as flat f32: [re0, im0, re1, im1, ...].
    Prints non-zero amplitudes in |k⟩ basis notation.
    """
    values = parse_iree_f32_vector(iree_output)
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
