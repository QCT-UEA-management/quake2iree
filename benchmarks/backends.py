# Compatibility shim — logic has moved to q2i/backends.py
from q2i.backends import *  # noqa: F401, F403
from q2i.backends import (
    IREEBackend, CUDAQBackend,
    IREE_CPU, IREE_VMVX, CUDAQ_CPU, CUDAQ_GPU, KNOWN_BACKENDS,
    iree_cuda, iree_rocm, iree_hip, iree_vulkan, iree_metal, resolve_backend,
)
