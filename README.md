# quake2iree

## Purpose and Scope

The quake2iree repository implements an MLIR dialect conversion system that transforms quantum programs written in the Quake dialect into IREE-compatible classical MLIR representations. This enables quantum algorithms to be executed on classical hardware through the IREE runtime system.

The system serves as a bridge between quantum computing frameworks (specifically CUDA-Q's Quake dialect) and classical machine learning compiler infrastructure (IREE), providing a complete transformation pipeline from quantum operations to executable classical code.

## Extending IREE with MLIR Dialects

Read the full guide here: <https://iree.dev/reference/extensions/#1-target-iree-input-dialects>

### Summary

The easiest, cleanest, and most robust way to extend IREE is by leveraging MLIR’s design for dialect composition and conversion. IREE supports multiple input dialects such as:

    * tosa
    * mhlo
    * linalg
    * arith, math, tensor, and scf

Any source IR that can be converted into this mix of dialects—either directly or transitively—will integrate seamlessly with the entire IREE pipeline across all deployment configurations and targets.

## Building the project

A convenience script is provided at the root of the repository:

```bash
chmod +x build.sh
./build.sh
```

This will create the `build/` directory, configure the project with CMake and Ninja, and compile it. The `q2i-opt` binary will be available at `build/tool/q2i-opt` once the build completes.

## Running the tests

The test suite uses Python's built-in `unittest` framework and lives in the `test/` directory. It requires the project to be built first (see above).

```bash
cd test
python3 -m unittest test_mlir_conversion -v
```

Each test runs the `q2i-opt --quake-to-standard` pass on a Quake dialect MLIR input file and verifies that the output is valid standard MLIR. A passing run looks like:

```
test_01_quake_alloca (test_mlir_conversion.TestMlirConversions) ... ok
test_02_quake_veq_size (test_mlir_conversion.TestMlirConversions) ... ok
test_03_quake_dealloc (test_mlir_conversion.TestMlirConversions) ... ok
test_04_quake_concat (test_mlir_conversion.TestMlirConversions) ... ok
test_05_quake_extractref (test_mlir_conversion.TestMlirConversions) ... ok
test_06_quake_initializestate (test_mlir_conversion.TestMlirConversions) ... ok
test_binary_exists_and_executable (test_mlir_conversion.TestQ2IOptAvailable) ... ok

----------------------------------------------------------------------
Ran 7 tests in 0.090s

OK
```

## Contribution Guidelines

Have first a look at our documentation on [dialect conversion](docs/how_to_convert.md)

Create a new branch for any feature or fix (do not commit directly to develop).

Submit your changes through a Pull Request.

Make sure all tests pass before opening a Pull Request.
