"""Example 10 — Parametric kernels with runtime angles.

Angles are passed as f64 function arguments rather than compile-time
constants. IREE receives them via --input=<value>::f64 at run time,
so the same compiled binary can be called with different angles.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cudaq
from examples.pipeline import run_statevector_kernel

_SV = dict(show_quake=False)


@cudaq.kernel
def rx_angle(theta: float):
    q = cudaq.qvector(1)
    rx(theta, q[0])


@cudaq.kernel
def ry_angle(theta: float):
    q = cudaq.qvector(1)
    ry(theta, q[0])


@cudaq.kernel
def rz_angle(lam: float):
    q = cudaq.qvector(1)
    h(q[0])
    rz(lam, q[0])


@cudaq.kernel
def r1_angle(lam: float):
    q = cudaq.qvector(1)
    h(q[0])
    r1(lam, q[0])


@cudaq.kernel
def ansatz(theta: float, phi: float):
    q = cudaq.qvector(2)
    ry(theta, q[0])
    rx(phi, q[1])
    cx(q[0], q[1])


ok = True

# RX(π)|0⟩ = -i|1⟩
ok = run_statevector_kernel(
    rx_angle, "RX(π) |0⟩ → -i|1⟩",
    kernel_args=[math.pi],
    expected={1: -1j},
    **_SV,
) and ok

# RY(π/2)|0⟩ = |+⟩ = (|0⟩+|1⟩)/√2
k = 1 / math.sqrt(2)
ok = run_statevector_kernel(
    ry_angle, "RY(π/2) |0⟩ → |+⟩",
    kernel_args=[math.pi / 2],
    expected={0: k, 1: k},
    **_SV,
) and ok

# H then RZ(π/2): |+⟩ → (|0⟩ - i|1⟩)/√2  (phase shift on |1⟩ by exp(iπ/4))
# RZ(λ)|+⟩: amplitude[0] *= exp(-iλ/2), amplitude[1] *= exp(iλ/2)
# λ=π/2 → amp[0] = (1/√2)*exp(-iπ/4) = (1-i)/(2), amp[1] = (1/√2)*exp(iπ/4) = (1+i)/(2)
ok = run_statevector_kernel(
    rz_angle, "H then RZ(π/2) |0⟩",
    kernel_args=[math.pi / 2],
    expected={0: 0.5 - 0.5j, 1: 0.5 + 0.5j},
    **_SV,
) and ok

# H then R1(π/3): R1(λ)|1⟩ = exp(iλ)|1⟩, R1(λ)|0⟩ = |0⟩
lam = math.pi / 3
ok = run_statevector_kernel(
    r1_angle, "H then R1(π/3) |0⟩",
    kernel_args=[lam],
    expected={
        0: k,
        1: complex(k * math.cos(lam), k * math.sin(lam)),
    },
    **_SV,
) and ok

# 2-qubit ansatz: RY(π/4)⊗RX(π/3) then CNOT
theta, phi = math.pi / 4, math.pi / 3
cy  = math.cos(theta / 2)   # cos(θ/2) — |0⟩ amplitude of q[0] after RY
cy_ = math.sin(theta / 2)   # sin(θ/2) — |1⟩ amplitude of q[0] after RY
cx_ = math.cos(phi / 2)     # cos(φ/2) — |0⟩ amplitude of q[1] after RX
sx_ = math.sin(phi / 2)     # sin(φ/2) — |1⟩ amplitude of q[1] after RX
# q[0] is LSB: k=0→q[0]=0,q[1]=0 / k=1→q[0]=1,q[1]=0 / k=2→q[0]=0,q[1]=1 / k=3→q[0]=1,q[1]=1
# Before CNOT: cy*cx_|00⟩ + cy_*cx_|01⟩ - i*cy*sx_|10⟩ - i*cy_*sx_|11⟩
# CNOT(ctrl=q[0], tgt=q[1]) swaps |01⟩↔|11⟩:
ok = run_statevector_kernel(
    ansatz, "RY(π/4)⊗RX(π/3), CNOT ansatz",
    kernel_args=[theta, phi],
    expected={
        0: complex(cy  * cx_,  0),          # |00⟩ unchanged
        1: complex(0,   -cy_ * sx_),        # |01⟩ ← old |11⟩
        2: complex(0,   -cy  * sx_),        # |10⟩ unchanged
        3: complex(cy_ * cx_,  0),          # |11⟩ ← old |01⟩
    },
    **_SV,
) and ok

print()
print("All parametric kernel tests passed." if ok else "Some tests FAILED.")
sys.exit(0 if ok else 1)
