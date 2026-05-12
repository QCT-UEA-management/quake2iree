"""Backend descriptors for IREE and CUDA-Q execution targets.

Each descriptor carries all the information needed to compile and run a kernel
on a specific device. Adding a new backend is a matter of defining a new
IREEBackend or CUDAQBackend instance — no changes to runner.py are needed.

IREE target-backend strings: llvm-cpu, cuda, rocm, vulkan-spirv, vmvx
IREE driver strings:          local-task, cuda, rocm, vulkan, local-task
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field


@dataclass(frozen=True)
class IREEBackend:
    """Describes an IREE compilation target and its matching runtime driver."""

    name: str
    target_backend: str
    driver: str
    extra_compile_args: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class CUDAQBackend:
    """Describes a CUDA-Q simulation target (passed to cudaq.set_target)."""

    name: str
    target: str


# ---------------------------------------------------------------------------
# CPU
# ---------------------------------------------------------------------------

IREE_CPU = IREEBackend(
    name="iree-cpu",
    target_backend="llvm-cpu",
    driver="local-task",
)

CUDAQ_CPU = CUDAQBackend(name="cudaq-cpu", target="qpp-cpu")

# ---------------------------------------------------------------------------
# NVIDIA CUDA
# ---------------------------------------------------------------------------

def _detect_cuda_sm() -> str:
    """Return the SM string for the first visible GPU, e.g. 'sm_75'."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        cap = r.stdout.strip().splitlines()[0].strip().replace(".", "")
        return f"sm_{cap}"
    except Exception:
        return "sm_80"


def iree_cuda(sm: str | None = None) -> IREEBackend:
    """IREE CUDA backend. Detects GPU SM version automatically when sm is None."""
    target = sm or _detect_cuda_sm()
    return IREEBackend(
        name="iree-cuda",
        target_backend="cuda",
        driver="cuda",
        extra_compile_args=(f"--iree-cuda-target={target}",),
    )


CUDAQ_GPU = CUDAQBackend(name="cudaq-gpu", target="nvidia")

# ---------------------------------------------------------------------------
# AMD ROCm  (future)
# ---------------------------------------------------------------------------

def iree_rocm(target_chip: str = "gfx1100") -> IREEBackend:
    """IREE ROCm backend for AMD GPUs."""
    return IREEBackend(
        name="iree-rocm",
        target_backend="rocm",
        driver="rocm",
        extra_compile_args=(f"--iree-rocm-target-chip={target_chip}",),
    )


# ---------------------------------------------------------------------------
# Vulkan  (future — cross-vendor GPU)
# ---------------------------------------------------------------------------

def iree_vulkan() -> IREEBackend:
    """IREE Vulkan backend for cross-vendor GPU execution."""
    return IREEBackend(
        name="iree-vulkan",
        target_backend="vulkan-spirv",
        driver="vulkan",
    )


# ---------------------------------------------------------------------------
# VMVX  (IREE portable reference interpreter)
# ---------------------------------------------------------------------------

IREE_VMVX = IREEBackend(
    name="iree-vmvx",
    target_backend="vmvx",
    driver="local-task",
)
