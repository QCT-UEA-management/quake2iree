"""Backend descriptors for IREE and CUDA-Q execution targets.

Each descriptor carries all the information needed to compile and run a kernel
on a specific device. Adding a new backend requires only two steps here:
  1. Define an IREEBackend constant or factory function.
  2. Add an entry to resolve_backend() and KNOWN_BACKENDS.
No changes to runner.py are needed.

IREE target-backend strings: llvm-cpu, cuda, rocm, vulkan-spirv, metal, vmvx
IREE driver strings:          local-task, cuda, hip, vulkan, metal, local-task
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
    extra_compile_args=("--iree-llvmcpu-target-cpu=host",),
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
# AMD ROCm
# ---------------------------------------------------------------------------

def iree_rocm(target_chip: str = "gfx1100") -> IREEBackend:
    """IREE ROCm/HIP backend for AMD GPUs.

    iree-compile target backend is still "rocm"; the runtime driver was
    renamed to "hip" in recent IREE releases.  The compile flag also changed:
    --iree-hip-target replaces the old --iree-rocm-target-chip.

    Pass the GFX chip string for your GPU, e.g. 'gfx906' (MI50),
    'gfx908' (MI100), 'gfx90a' (MI250), 'gfx1100' (RX 7900 / RDNA3).
    """
    return IREEBackend(
        name="iree-rocm",
        target_backend="rocm",
        driver="hip",
        extra_compile_args=(f"--iree-hip-target={target_chip}",),
    )


# ---------------------------------------------------------------------------
# Vulkan  (cross-vendor GPU: NVIDIA, AMD, Intel, via MoltenVK on macOS)
# ---------------------------------------------------------------------------

def iree_vulkan() -> IREEBackend:
    """IREE Vulkan backend for cross-vendor GPU execution.

    On macOS this runs via MoltenVK, which translates Vulkan to Metal.
    """
    return IREEBackend(
        name="iree-vulkan",
        target_backend="vulkan-spirv",
        driver="vulkan",
    )


# ---------------------------------------------------------------------------
# Metal  (Apple GPU — macOS / iOS native)
# ---------------------------------------------------------------------------

def iree_metal() -> IREEBackend:
    """IREE Metal backend for Apple GPU (macOS / iOS)."""
    return IREEBackend(
        name="iree-metal",
        target_backend="metal",
        driver="metal",
    )


# ---------------------------------------------------------------------------
# VMVX  (IREE portable reference interpreter — always available, slowest)
# ---------------------------------------------------------------------------

IREE_VMVX = IREEBackend(
    name="iree-vmvx",
    target_backend="vmvx",
    driver="local-task",
)

# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

#: All backend names accepted by the --backends CLI flag.
KNOWN_BACKENDS: frozenset[str] = frozenset({
    "iree-cpu",
    "iree-cuda",
    "iree-rocm",
    "iree-vulkan",
    "iree-metal",
    "iree-vmvx",
    "cudaq-cpu",
    "cudaq-gpu",
})


def resolve_backend(name: str) -> IREEBackend | CUDAQBackend:
    """Return the backend descriptor for *name*.

    This is the single place to extend when adding a new backend target.
    Raises KeyError for unknown names.
    """
    match name:
        case "iree-cpu":     return IREE_CPU
        case "iree-cuda":    return iree_cuda()
        case "iree-rocm":    return iree_rocm()
        case "iree-vulkan":  return iree_vulkan()
        case "iree-metal":   return iree_metal()
        case "iree-vmvx":    return IREE_VMVX
        case "cudaq-cpu":    return CUDAQ_CPU
        case "cudaq-gpu":    return CUDAQ_GPU
        case _:              raise KeyError(f"Unknown backend: {name!r}")
