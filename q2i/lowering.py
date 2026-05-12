"""Core lowering pipeline: CUDA-Q kernel → quake MLIR → standard dialects.

These helpers are shared by both examples/ and benchmarks/. They cover the
stage that is specific to quake2iree (quake dialect removal); IREE compilation
and execution are handled separately by each consumer.
"""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent
Q2I_OPT = ROOT / "build" / "tool" / "q2i-opt"


def emit_quake(kernel) -> str:
    """Compile a CUDA-Q kernel and return its raw quake MLIR string."""
    kernel.compile()
    return str(kernel.qkeModule)


def entrypoint_name(quake_ir: str) -> str | None:
    """Return the symbol name of the cudaq-entrypoint function, or None."""
    m = re.search(r'func\.func @(\S+?)\(.*?"cudaq-entrypoint"', quake_ir)
    return m.group(1) if m else None


def strip_cudaq_run_wrappers(quake_ir: str) -> str:
    """Remove CUDA-Q .run helper functions that contain unsupported cc ops."""
    output = []
    skipping = False
    brace_depth = 0

    for line in quake_ir.splitlines():
        starts_run_wrapper = (
            "func.func @" in line
            and (".run()" in line or ".run.entry()" in line)
        )
        if not skipping and starts_run_wrapper:
            skipping = True
            brace_depth = line.count("{") - line.count("}")
            if brace_depth <= 0:
                skipping = False
            continue

        if skipping:
            brace_depth += line.count("{") - line.count("}")
            if brace_depth <= 0:
                skipping = False
            continue

        output.append(line)

    return "\n".join(output) + "\n"


def q2i_convert(input_file: Path, output_file: Path) -> subprocess.CompletedProcess:
    """Lower quake dialect to standard dialects via q2i-opt."""
    return subprocess.run(
        [str(Q2I_OPT), str(input_file), "--quake-to-standard", "-o", str(output_file)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
