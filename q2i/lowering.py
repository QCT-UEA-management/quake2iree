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


def lower(kernel) -> tuple[str, str]:
    """Lower a CUDA-Q kernel to standard MLIR.

    Convenience wrapper: emit_quake → strip → q2i_convert.
    Returns (mlir_text, func_name). Raises RuntimeError on failure.
    """
    import tempfile

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


def q2i_convert(input_file: Path, output_file: Path) -> subprocess.CompletedProcess:
    """Lower quake dialect to standard dialects via q2i-opt.

    Post-processes the output to add `output_shape` to `tensor.expand_shape`
    ops — required by iree-compile (MLIR 20) but absent in the MLIR 16 text
    format that q2i-opt emits.
    """
    result = subprocess.run(
        [str(Q2I_OPT), str(input_file), "--quake-to-standard", "-o", str(output_file)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode == 0 and output_file.exists():
        _patch_expand_shape(output_file)
    return result


def _patch_expand_shape(mlir_file: Path) -> None:
    """Rewrite tensor.expand_shape to include the output_shape keyword.

    q2i-opt is compiled against MLIR 16 which omits output_shape in the
    textual format. iree-compile (MLIR 20) requires it. The shape values
    are derived from the destination tensor type, which is always static
    in our lowering.
    """
    text = mlir_file.read_text()

    def rewrite(m: re.Match) -> str:
        op_reassoc = m.group(1)   # "tensor.expand_shape %X [[...]]"
        src_type   = m.group(2)   # e.g. "8xf32"
        dest_type  = m.group(3)   # e.g. "2x2x1x2xf32"
        # All 'x'-separated parts except the last are static dimensions.
        parts = dest_type.split('x')
        dims = parts[:-1]
        output_shape = '[' + ', '.join(dims) + ']'
        return (f'{op_reassoc} output_shape {output_shape} '
                f': tensor<{src_type}> into tensor<{dest_type}>')

    patched = re.sub(
        r'(tensor\.expand_shape\s+\S+\s+\[\[(?:(?!\]\]).)+\]\])'
        r'\s*:\s*tensor<([^>]+)>\s*into\s*tensor<([^>]+)>',
        rewrite,
        text,
    )
    mlir_file.write_text(patched)
