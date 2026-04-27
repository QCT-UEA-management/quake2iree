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
