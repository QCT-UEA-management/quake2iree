# NVIDIA CUDA-Q: Quake Dialect Ops

The **Quake dialect** is NVIDIA CUDA-Q's MLIR dialect for quantum kernel representation. It comes in two flavors: **memory semantics** (using `ref`/`veq` types, mutable in place) and **value semantics** (using `wire`/`control` types, following linear type discipline for circuit optimization).

---

## Allocation & Deallocation

| Op | Description |
|---|---|
| [`quake.alloca`](https://github.com/QCT-UEA-management/quake2iree/blob/b351e9ec9c63bb046ef54ff51ce0eadf84b7d626/Conversion/QuakeToStandard.cpp#L65) | Allocate qubits (single `!quake.ref` or vector `!quake.veq<N>`) |
| [`quake.dealloc`](https://github.com/QCT-UEA-management/quake2iree/blob/b351e9ec9c63bb046ef54ff51ce0eadf84b7d626/Conversion/QuakeToStandard.cpp#L47) | Deallocate qubits |
| [`quake.init_state`](https://github.com/QCT-UEA-management/quake2iree/blob/b351e9ec9c63bb046ef54ff51ce0eadf84b7d626/Conversion/QuakeToStandard.cpp#L115) | Initialize qubits to a specific quantum state |

---

## Vector Manipulation

| Op | Description |
|---|---|
| [`quake.concat`](https://github.com/QCT-UEA-management/quake2iree/blob/b351e9ec9c63bb046ef54ff51ce0eadf84b7d626/Conversion/QuakeToStandard.cpp#L137) | Concatenate quantum refs/vectors into a larger vector |
| [`quake.extract_ref`](https://github.com/QCT-UEA-management/quake2iree/blob/b351e9ec9c63bb046ef54ff51ce0eadf84b7d626/Conversion/QuakeToStandard.cpp#L280) | Extract a single `!quake.ref` from a `!quake.veq`, by constant or dynamic index |
| `quake.subveq` | Extract a subvector from a `!quake.veq` |
| `quake.relax_size` | Convert a statically-sized `!quake.veq<N>` to a dynamically-sized `!quake.veq<?>` |
| [`quake.veq_size`](https://github.com/QCT-UEA-management/quake2iree/blob/b351e9ec9c63bb046ef54ff51ce0eadf84b7d626/Conversion/QuakeToStandard.cpp#L264) | Get the runtime size of a qubit vector |

---

## Quantum Gate Operations

All gates support optional control qubits and an adjoint (`is_adj`) flag.

| Op | Gate |
|---|---|
| `quake.h` | Hadamard |
| `quake.x` | Pauli-X (also used as CNOT with controls) |
| `quake.y` | Pauli-Y |
| `quake.z` | Pauli-Z |
| `quake.s` | S gate |
| `quake.t` | T gate |
| `quake.rx` | Rotation around X (1 angle parameter) |
| `quake.ry` | Rotation around Y (1 angle parameter) |
| `quake.rz` | Rotation around Z (1 angle parameter) |
| `quake.r1` | Phase rotation (1 angle parameter) |
| `quake.swap` | Swap two qubits |

---

## Measurement Operations

Measurement ops produce `!quake.measure` results.

| Op | Description |
|---|---|
| `quake.mz` | Measure in the Z (computational) basis |
| `quake.mx` | Measure in the X basis |
| `quake.my` | Measure in the Y basis |
| `quake.discriminate` | Convert a `!quake.measure` result to a classical boolean |

---

## Application & Control Flow

| Op | Description |
|---|---|
| `quake.apply` | Apply a user-defined kernel (supports direct symbol references or indirect callables, with optional controls and adjoint) |
| `quake.compute_action` | Encode the compute-action-uncompute idiom (the uncompute is automatically generated as the adjoint) |

---

## Wire / Value-Semantics Ops

For the value-semantics form of Quake using linear `!quake.wire` types that must be used exactly once, enforcing the no-cloning theorem.

| Op | Description |
|---|---|
| `quake.null_wire` | Create an initial wire state (\|0⟩) |
| `quake.sink` | Consume a wire at the end of a circuit |
| `quake.unwrap` | Convert a `!quake.ref` to a `!quake.wire` |
| `quake.wrap` | Convert a `!quake.wire` back to a `!quake.ref` |

---

## Type System

| Type | Syntax | Description |
|---|---|---|
| `RefType` | `!quake.ref` | Single mutable qubit reference (memory semantics) |
| `VeqType` | `!quake.veq<N>` or `!quake.veq<?>` | Static or dynamic qubit vector |
| `WireType` | `!quake.wire` | Linear quantum value (value semantics) |
| `ControlType` | `!quake.control` | Control qubit marker |
| `MeasureType` | `!quake.measure` | Opaque measurement result |
| `StruqType` | `!quake.struq<...>` | Structured/aggregate quantum type |
