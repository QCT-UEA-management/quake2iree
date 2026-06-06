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
chmod +x scripts/build.sh
./scripts/build.sh
```

This will create the `build/` directory, configure the project with CMake and Ninja, and compile it. The `q2i-opt` binary will be available at `build/tool/q2i-opt` once the build completes.

## Running the tests

The test suite uses [pytest](https://pytest.org) and lives in
`test/test_circuits.py`. It drives real CUDA-Q kernels through the full
conversion pipeline and checks statevector outputs and sampling distributions.

Build the project first, then run:

```bash
python3 -m pytest test/test_circuits.py -v
```

For a quieter run (just pass/fail summary):

```bash
python3 -m pytest test/test_circuits.py -q
```

There are 20 tests across 8 classes covering Bell state, GHZ, rotations, phase gates, SWAP, measurements, reset, controlled-Z, and sampling.

## Pipeline examples

The `examples/` directory contains end-to-end scripts that drive real CUDA-Q
kernels through the full conversion pipeline. Unlike the test suite, examples
are expected to fail at the current frontier of the implementation — each
failure surfaces a concrete missing piece in `quake2iree`.

### Shared helper — `examples/pipeline.py`

Provides thin wrappers around each pipeline stage and display utilities used
by all example scripts:

| Function | Stage |
|---|---|
| `emit_quake(kernel)` | Compile a CUDA-Q kernel and return its raw quake MLIR string |
| `entrypoint_name(quake_ir)` | Extract the `cudaq-entrypoint` function symbol name |
| `q2i_convert(input, output)` | Run `q2i-opt --quake-to-standard` |
| `iree_compile(input, output)` | Run `iree-compile --iree-hal-target-backends=llvm-cpu` |
| `iree_run(vmfb, function)` | Run `iree-run-module` |

### Running an example

```bash
python3 examples/01_bell.py
```

Each script prints the intermediate IR at every stage and stops with a
descriptive error at the first failure, so you can immediately see where the pipeline breaks and why.

## Contributors

* Mario Hernandez Vera — <mario.hernandezvera@lrz.de>
* Marco De Pascal — <marco.depascale@munich-quantum-valley.de>

## Contribution Guidelines

Have first a look at our documentation on [dialect conversion](docs/how_to_convert.md)

Create a new branch for any feature or fix (do not commit directly to develop).

Submit your changes through a Pull Request.

Make sure all tests pass before opening a Pull Request.
