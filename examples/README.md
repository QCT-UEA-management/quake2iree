# quake2iree examples

This directory contains small executable examples that drive real CUDA-Q
kernels through the current quake2iree pipeline:

```text
CUDA-Q kernel -> quake MLIR -> q2i-opt -> IREE compile -> IREE run
```

The examples are the main development frontier for now. Each script should
exercise one small, concrete feature and should fail at the first unsupported
pipeline stage with enough output to explain what is missing.

## Structure

Shared plumbing lives in `pipeline.py`:

- `emit_quake` emits CUDA-Q's quake MLIR.
- `q2i_convert` runs `q2i-opt --quake-to-standard`.
- `iree_compile` builds an IREE VMFB.
- `iree_run` executes the compiled module.
- `run_statevector_kernel` runs the common statevector-oriented workflow used
  by the example scripts.
- `check_statevector` optionally checks sparse expected amplitudes.
- `run_i1_kernel` runs kernels that return one classical `i1` result.

Individual example scripts should stay focused on kernels and expectations.
Avoid duplicating pipeline mechanics inside each script.

## Current role

These scripts are intentionally both demos and development checks. They print
intermediate IR and final statevectors so failures are easy to diagnose while
the lowering is still growing.

Later, the same cases can become stricter tests by adding assertion helpers for
expected amplitudes and, once measurement is supported, expected classical
results.
