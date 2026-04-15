# quake2iree

## Purpose and Scope

The quake2iree repository implements an MLIR dialect conversion system that transforms quantum programs written in the Quake dialect into IREE-compatible classical MLIR representations. This enables quantum algorithms to be executed on classical hardware through the IREE runtime system.

The system serves as a bridge between quantum computing frameworks (specifically CUDA-Q's Quake dialect) and classical machine learning compiler infrastructure (IREE), providing a complete transformation pipeline from quantum operations to executable classical code.

## Extending IREE with MLIR Dialects

Read the full guide here: <https://iree.dev/reference/extensions/#1-target-iree-input-dialects>

### Summary

The easiest, cleanest, and most robust way to extend IREE is by leveraging MLIR's design for dialect composition and conversion. IREE supports multiple input dialects such as:

* tosa
* mhlo
* linalg
* arith, math, tensor, and scf

Any source IR that can be converted into this mix of dialects, either directly or transitively, will integrate seamlessly with the entire IREE pipeline across all deployment configurations and targets.

## Building the project

A convenience script is provided at the root of the repository:

```bash
chmod +x build.sh
./build.sh
```

This will create the `build/` directory, configure the project with CMake and Ninja, and compile it. The `q2i-opt` binary will be available at `build/tool/q2i-opt` once the build completes.

## Running the tests

The test suite uses Python's built-in `unittest` framework and lives in the `test/` directory. MLIR test inputs live in `test/data/` so the test modules stay separate from the sample IR files.

The recommended way to run all tests is from the repository root:

```bash
./test.sh
```

This script expects the project to be built first with `./build.sh`. It runs unittest discovery over the `test/` directory and disables Python bytecode writes so test runs do not update `__pycache__` files.

There are currently two test modules:

* `test/test_mlir_conversion.py` runs `q2i-opt --quake-to-standard` on Quake dialect MLIR inputs from `test/data/` and checks that conversion succeeds.
* `test/test_iree_compile.py` checks that lowered MLIR inputs from `test/data/` compile with `iree-compile`, and also checks the end-to-end path from Quake input to lowered MLIR to IREE bytecode.

The IREE tests require `iree-compile` to be available in `PATH`. If it is not installed, those tests are skipped by unittest.

You can also run individual test modules directly:

```bash
cd test
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest test_mlir_conversion -v
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest test_iree_compile -v
```

## Contribution Guidelines

Have first a look at our documentation on [dialect conversion](docs/how_to_convert.md)

Create a new branch for any feature or fix (do not commit directly to develop).

Submit your changes through a Pull Request.

Make sure all tests pass before opening a Pull Request.
