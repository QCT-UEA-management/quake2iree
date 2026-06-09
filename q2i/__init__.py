# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

"""q2i — CUDA-Q to IREE pipeline.

Five-stage API::

    # Stage 1: CUDA-Q kernel → standard MLIR
    mlir_text, func_name = q2i.lower(kernel)

    # Stage 2: MLIR → VMFB flatbuffer (in-process, no subprocess)
    vmfb = q2i.compile(mlir_text, backend=q2i.IREE_CPU)

    # Stage 3: execute
    sv   = q2i.run_statevector(vmfb, func_name, n_qubits=2)
    bit  = q2i.run_i1(vmfb, func_name, n_qubits=1)

High-level pipeline runners (lower + compile + run + check in one call)::

    ok = q2i.run_statevector_kernel(kernel, "label", expected={0: k, 3: k})
    ok = q2i.run_i1_kernel(kernel, "label", expected=False)
    ok = q2i.run_sample_kernel(kernel, "label", shots=1000, expected_keys={"00","11"})

Backends::

    q2i.IREE_CPU, q2i.IREE_VMVX
    q2i.iree_cuda(), q2i.iree_rocm(), q2i.iree_hip()

Benchmarking::

    result = q2i.TimingResult(...)
    q2i.benchmark_cli("MyCircuit", make_kernel_factory)
"""

# Stage 1 – lowering
from q2i.lowering import (
    lower,
    emit_quake,
    entrypoint_name,
    strip_cudaq_run_wrappers,
    q2i_convert,
)

# Stage 2 – compilation
from q2i.compile import compile_mlir as compile

# Stage 3 – execution
from q2i.run import (
    run_statevector,
    run_i1,
    sample_from_statevector,
    # display helpers
    banner,
    show_ir,
    step_ok,
    step_fail,
    print_statevector,
    check_statevector,
    check_sample_counts,
    # full-pipeline runners
    run_statevector_kernel,
    run_i1_kernel,
    run_sample_kernel,
)

# Backends
from q2i.backends import (
    IREEBackend,
    CUDAQBackend,
    IREE_CPU,
    IREE_VMVX,
    CUDAQ_CPU,
    CUDAQ_GPU,
    KNOWN_BACKENDS,
    iree_cuda,
    iree_rocm,
    iree_hip,
    iree_vulkan,
    iree_metal,
    resolve_backend,
)

# Benchmarking
from q2i.benchmark import (
    TimingResult,
    compile_for_iree,
    time_iree,
    time_cudaq,
    run_sweep,
    save_csv,
    benchmark_cli,
)

__all__ = [
    # lowering
    "lower", "emit_quake", "entrypoint_name", "strip_cudaq_run_wrappers", "q2i_convert",
    # compilation
    "compile",
    # execution
    "run_statevector", "run_i1", "sample_from_statevector",
    "run_statevector_kernel", "run_i1_kernel", "run_sample_kernel",
    # display
    "banner", "show_ir", "step_ok", "step_fail",
    "print_statevector", "check_statevector", "check_sample_counts",
    # backends
    "IREEBackend", "CUDAQBackend",
    "IREE_CPU", "IREE_VMVX", "CUDAQ_CPU", "CUDAQ_GPU", "KNOWN_BACKENDS",
    "iree_cuda", "iree_rocm", "iree_hip", "iree_vulkan", "iree_metal", "resolve_backend",
    # benchmarking
    "TimingResult", "compile_for_iree", "time_iree", "time_cudaq",
    "run_sweep", "save_csv", "benchmark_cli",
]
