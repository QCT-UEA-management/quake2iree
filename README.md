# quake2iree

## Purpose and Scope

The quake2iree repository implements an MLIR dialect conversion system that transforms quantum programs written in the Quake dialect into IREE-compatible classical MLIR representations. This enables quantum algorithms to be executed on classical hardware through the IREE runtime system.

The system serves as a bridge between quantum computing frameworks (specifically CUDA-Q's Quake dialect) and classical machine learning compiler infrastructure (IREE), providing a complete transformation pipeline from quantum operations to executable classical code.

## Compile, link, and build your project

```bash
cd build
cmake -G Ninja ..
ninja -j4
```
