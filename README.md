# quake2iree

## Purpose and Scope

The quake2iree repository implements an MLIR dialect conversion system that transforms quantum programs written in the Quake dialect into IREE-compatible classical MLIR representations. This enables quantum algorithms to be executed on classical hardware through the IREE runtime system.

The system serves as a bridge between quantum computing frameworks (specifically CUDA-Q's Quake dialect) and classical machine learning compiler infrastructure (IREE), providing a complete transformation pipeline from quantum operations to executable classical code.

## Extending IREE with MLIR Dialects

Read the full guide here: <https://iree.dev/reference/extensions/#1-target-iree-input-dialects>

Summary
The easiest, cleanest, and most robust way to extend IREE is by leveraging MLIR’s design for dialect composition and conversion. IREE supports multiple input dialects such as:

    * tosa
    * mhlo
    * linalg
    * arith, math, tensor, and scf

Any source IR that can be converted into this mix of dialects—either directly or transitively—will integrate seamlessly with the entire IREE pipeline across all deployment configurations and targets.

## Compile, link, and build your project

```bash
cd build
cmake -G Ninja ..
ninja -j4
```
