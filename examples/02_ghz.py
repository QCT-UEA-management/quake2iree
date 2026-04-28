#!/usr/bin/env python3
"""GHZ state — three-qubit maximally entangled state.

Circuit:
    H(q[0])
    CNOT(ctrl=q[0], tgt=q[1])
    CNOT(ctrl=q[1], tgt=q[2])

Expected statevector: (|000⟩ + |111⟩) / √2

Full pipeline:
  CUDA-Q kernel  ->  quake MLIR  ->  q2i-opt (quake-to-standard)
  ->  iree-compile  ->  iree-run-module

Run from the repo root:
  python examples/02_ghz.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cudaq
from examples.pipeline import (
    banner,
    emit_quake,
    entrypoint_name,
    iree_compile,
    iree_run,
    print_statevector,
    q2i_convert,
    show_ir,
    step_fail,
    step_ok,
)


# ---------------------------------------------------------------------------
# Kernel definition
# ---------------------------------------------------------------------------

@cudaq.kernel
def ghz():
    q = cudaq.qvector(3)
    h(q[0])
    cx(q[0], q[1])
    cx(q[1], q[2])


# ---------------------------------------------------------------------------
# Stage 1 — emit quake IR from CUDA-Q
# ---------------------------------------------------------------------------

banner("Stage 1: CUDA-Q  ->  quake MLIR")

quake_ir = emit_quake(ghz)
func_name = entrypoint_name(quake_ir)

show_ir("raw quake IR from CUDA-Q", quake_ir)
step_ok("emit_quake", f"entrypoint = {func_name}")

# ---------------------------------------------------------------------------
# Stage 2 — quake-to-standard via q2i-opt
# ---------------------------------------------------------------------------

banner("Stage 2: quake MLIR  ->  standard dialects  (q2i-opt)")

with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    quake_file   = tmp_path / "ghz.mlir"
    lowered_file = tmp_path / "ghz_lowered.mlir"
    vmfb_file    = tmp_path / "ghz.vmfb"

    quake_file.write_text(quake_ir)
    result = q2i_convert(quake_file, lowered_file)

    if result.returncode != 0:
        step_fail("q2i-opt --quake-to-standard", result.stderr)
        print("\n  Pipeline stopped here — fix q2i-opt errors above to continue.")
        sys.exit(1)

    lowered_ir = lowered_file.read_text()
    show_ir("lowered MLIR (standard dialects)", lowered_ir)
    step_ok("q2i-opt --quake-to-standard")

    # -----------------------------------------------------------------------
    # Stage 3 — iree-compile
    # -----------------------------------------------------------------------

    banner("Stage 3: standard dialects  ->  IREE vmfb  (iree-compile)")

    result = iree_compile(lowered_file, vmfb_file)

    if result.returncode != 0:
        step_fail("iree-compile", result.stderr)
        print("\n  Pipeline stopped here — fix iree-compile errors above to continue.")
        sys.exit(1)

    step_ok("iree-compile", f"{vmfb_file.stat().st_size} bytes")

    # -----------------------------------------------------------------------
    # Stage 4 — iree-run-module
    # -----------------------------------------------------------------------

    banner(f"Stage 4: execute IREE module  (function = {func_name})")

    result = iree_run(vmfb_file, func_name)

    if result.returncode != 0:
        step_fail("iree-run-module", result.stderr)
        sys.exit(1)

    step_ok("iree-run-module")
    print(f"\n  raw output:\n  {result.stdout.strip()}")
    print()
    print_statevector(result.stdout)

banner("Done")
