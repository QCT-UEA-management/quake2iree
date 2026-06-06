# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

"""IREE in-process compilation: lowered MLIR → platform-specific VMFB.

Uses iree.compiler Python bindings — no subprocess or temp files required.
This replaces the former subprocess iree-compile call in examples/pipeline.py.
"""

import iree.compiler as irec

from q2i.backends import IREEBackend, IREE_CPU


def compile_mlir(mlir_text: str, backend: IREEBackend = IREE_CPU) -> bytes:
    """Compile lowered MLIR to a VMFB flatbuffer for the given backend.

    Returns raw vmfb bytes. Raises on compilation failure (irec raises
    directly with the compiler diagnostics as the exception message).
    """
    return irec.compile_str(
        mlir_text,
        target_backends=[backend.target_backend],
        extra_args=list(backend.extra_compile_args),
    )
