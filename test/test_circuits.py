# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

"""
Full-pipeline regression tests — mirrors examples/01–10.

Kernels are defined at module level (cudaq.kernel requires source inspection).
Pipeline helpers are called with show_quake=False so pytest output stays clean;
captured output is only surfaced when a test fails, providing debug context.

Sampling tests use a fixed seed for deterministic counts across runs.
Parametric tests pass runtime f64 angles via kernel_args.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cudaq
from q2i import (
    run_i1_kernel,
    run_sample_kernel,
    run_statevector_kernel,
)

_SV = dict(show_quake=False)
_I1 = dict(show_quake=False)
_SA = dict(show_quake=False, seed=42)

# ---------------------------------------------------------------------------
# Kernels — all at module level (cudaq.kernel source inspection requirement)
# ---------------------------------------------------------------------------

# 01 Bell
@cudaq.kernel
def bell():
    q = cudaq.qvector(2)
    h(q[0])
    cx(q[0], q[1])


# 02 GHZ
@cudaq.kernel
def ghz():
    q = cudaq.qvector(3)
    h(q[0])
    cx(q[0], q[1])
    cx(q[1], q[2])


# 03 Rotations
@cudaq.kernel
def rx_half_turn():
    q = cudaq.qvector(1)
    rx(math.pi, q[0])


@cudaq.kernel
def ry_quarter():
    q = cudaq.qvector(1)
    ry(math.pi / 2, q[0])


@cudaq.kernel
def h_then_rz():
    q = cudaq.qvector(1)
    h(q[0])
    rz(math.pi / 2, q[0])


# 04 Phase gates
@cudaq.kernel
def h_then_s():
    q = cudaq.qvector(1)
    h(q[0])
    s(q[0])


@cudaq.kernel
def h_then_t():
    q = cudaq.qvector(1)
    h(q[0])
    t(q[0])


@cudaq.kernel
def h_then_r1():
    q = cudaq.qvector(1)
    h(q[0])
    r1(math.pi / 3, q[0])


# 05 SWAP
@cudaq.kernel
def x_first_then_swap():
    q = cudaq.qvector(2)
    x(q[0])
    swap(q[0], q[1])


@cudaq.kernel
def superposition_swap():
    q = cudaq.qvector(2)
    h(q[0])
    swap(q[0], q[1])


# 06 Measurements
@cudaq.kernel
def measure_zero() -> bool:
    q = cudaq.qvector(1)
    return mz(q[0])


@cudaq.kernel
def measure_one() -> bool:
    q = cudaq.qvector(1)
    x(q[0])
    return mz(q[0])


@cudaq.kernel
def measure_plus() -> bool:
    q = cudaq.qvector(1)
    h(q[0])
    return mz(q[0])


# 07 Reset
@cudaq.kernel
def reset_one():
    q = cudaq.qvector(1)
    x(q[0])
    reset(q[0])


@cudaq.kernel
def reset_plus():
    q = cudaq.qvector(1)
    h(q[0])
    reset(q[0])


# 08 Controlled-Z
@cudaq.kernel
def cz_on_11():
    q = cudaq.qvector(2)
    x(q[0])
    x(q[1])
    cz(q[0], q[1])


@cudaq.kernel
def cz_on_superposition():
    q = cudaq.qvector(2)
    h(q[0])
    h(q[1])
    cz(q[0], q[1])


# 09 Sampling
@cudaq.kernel
def uniform_1q_sample():
    q = cudaq.qvector(1)
    h(q[0])
    mz(q)


@cudaq.kernel
def bell_sample():
    q = cudaq.qvector(2)
    h(q[0])
    cx(q[0], q[1])
    mz(q)


@cudaq.kernel
def ghz_sample():
    q = cudaq.qvector(3)
    h(q[0])
    cx(q[0], q[1])
    cx(q[1], q[2])
    mz(q)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBell:
    def test_statevector(self):
        k = 1 / math.sqrt(2)
        assert run_statevector_kernel(bell, "bell", expected={0: k, 3: k}, **_SV)


class TestGHZ:
    def test_statevector(self):
        k = 1 / math.sqrt(2)
        assert run_statevector_kernel(ghz, "ghz", expected={0: k, 7: k}, **_SV)


class TestRotations:
    def test_rx_pi(self):
        assert run_statevector_kernel(rx_half_turn, "rx_half_turn", expected={1: -1j}, **_SV)

    def test_ry_pi_over_2(self):
        k = 1 / math.sqrt(2)
        assert run_statevector_kernel(ry_quarter, "ry_quarter", expected={0: k, 1: k}, **_SV)

    def test_h_then_rz_pi_over_2(self):
        assert run_statevector_kernel(
            h_then_rz, "h_then_rz", expected={0: 0.5 - 0.5j, 1: 0.5 + 0.5j}, **_SV
        )


class TestPhaseGates:
    def test_h_then_s(self):
        k = 1 / math.sqrt(2)
        assert run_statevector_kernel(h_then_s, "h_then_s", expected={0: k, 1: 1j * k}, **_SV)

    def test_h_then_t(self):
        k = 1 / math.sqrt(2)
        assert run_statevector_kernel(h_then_t, "h_then_t", expected={0: k, 1: 0.5 + 0.5j}, **_SV)

    def test_h_then_r1_pi_over_3(self):
        k = 1 / math.sqrt(2)
        assert run_statevector_kernel(
            h_then_r1, "h_then_r1",
            expected={0: k, 1: 0.5 / math.sqrt(2) + 0.5j * math.sqrt(3 / 2)},
            **_SV,
        )


class TestSwap:
    def test_x_swap(self):
        assert run_statevector_kernel(x_first_then_swap, "x_first_then_swap", expected={2: 1.0}, **_SV)

    def test_superposition_swap(self):
        k = 1 / math.sqrt(2)
        assert run_statevector_kernel(
            superposition_swap, "superposition_swap", expected={0: k, 2: k}, **_SV
        )


class TestMeasurements:
    def test_measure_zero(self):
        assert run_i1_kernel(measure_zero, "measure_zero", expected=False, **_I1)

    def test_measure_one(self):
        assert run_i1_kernel(measure_one, "measure_one", expected=True, **_I1)

    def test_measure_plus(self):
        assert run_i1_kernel(measure_plus, "measure_plus", expected=False, **_I1)


class TestReset:
    def test_reset_from_one(self):
        assert run_statevector_kernel(reset_one, "reset_one", expected={0: 1.0}, **_SV)

    def test_reset_from_plus(self):
        assert run_statevector_kernel(reset_plus, "reset_plus", expected={0: 1.0}, **_SV)


class TestControlledZ:
    def test_cz_on_11(self):
        assert run_statevector_kernel(cz_on_11, "cz_on_11", expected={3: -1.0}, **_SV)

    def test_cz_on_superposition(self):
        assert run_statevector_kernel(
            cz_on_superposition, "cz_on_superposition",
            expected={0: 0.5, 1: 0.5, 2: 0.5, 3: -0.5},
            **_SV,
        )


class TestSampling:
    def test_uniform_1q(self):
        assert run_sample_kernel(
            uniform_1q_sample, "uniform_1q", 1000, expected_keys={"0", "1"}, **_SA
        )

    def test_bell_sample(self):
        assert run_sample_kernel(
            bell_sample, "bell_sample", 1000, expected_keys={"00", "11"}, **_SA
        )

    def test_ghz_sample(self):
        assert run_sample_kernel(
            ghz_sample, "ghz_sample", 1000, expected_keys={"000", "111"}, **_SA
        )


# 10 Parametric kernels

@cudaq.kernel
def rx_param(theta: float):
    q = cudaq.qvector(1)
    rx(theta, q[0])


@cudaq.kernel
def ry_param(theta: float):
    q = cudaq.qvector(1)
    ry(theta, q[0])


@cudaq.kernel
def rz_param(lam: float):
    q = cudaq.qvector(1)
    h(q[0])
    rz(lam, q[0])


@cudaq.kernel
def r1_param(lam: float):
    q = cudaq.qvector(1)
    h(q[0])
    r1(lam, q[0])


@cudaq.kernel
def ansatz_param(theta: float, phi: float):
    q = cudaq.qvector(2)
    ry(theta, q[0])
    rx(phi, q[1])
    cx(q[0], q[1])


class TestParametric:
    def test_rx_pi(self):
        assert run_statevector_kernel(
            rx_param, "rx_param(π)",
            kernel_args=[math.pi],
            expected={1: -1j},
            **_SV,
        )

    def test_ry_half_pi(self):
        k = 1 / math.sqrt(2)
        assert run_statevector_kernel(
            ry_param, "ry_param(π/2)",
            kernel_args=[math.pi / 2],
            expected={0: k, 1: k},
            **_SV,
        )

    def test_rz_half_pi(self):
        assert run_statevector_kernel(
            rz_param, "rz_param(π/2)",
            kernel_args=[math.pi / 2],
            expected={0: 0.5 - 0.5j, 1: 0.5 + 0.5j},
            **_SV,
        )

    def test_r1_third_pi(self):
        k = 1 / math.sqrt(2)
        lam = math.pi / 3
        assert run_statevector_kernel(
            r1_param, "r1_param(π/3)",
            kernel_args=[lam],
            expected={0: k, 1: complex(k * math.cos(lam), k * math.sin(lam))},
            **_SV,
        )

    def test_ansatz_two_params(self):
        theta, phi = math.pi / 4, math.pi / 3
        cy  = math.cos(theta / 2)
        cy_ = math.sin(theta / 2)
        cx_ = math.cos(phi / 2)
        sx_ = math.sin(phi / 2)
        assert run_statevector_kernel(
            ansatz_param, "ansatz_param(π/4, π/3)",
            kernel_args=[theta, phi],
            expected={
                0: complex(cy  * cx_,  0),
                1: complex(0,  -cy_ * sx_),
                2: complex(0,  -cy  * sx_),
                3: complex(cy_ * cx_,  0),
            },
            **_SV,
        )
