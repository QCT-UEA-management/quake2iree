# quake2iree


## Purpose and Scope

The quake2iree repository implements an MLIR dialect conversion system that transforms quantum programs written in the Quake dialect into IREE-compatible classical MLIR representations. This enables quantum algorithms to be executed on classical hardware through the IREE runtime system.


## Current Implementation 

The current implementation of the quake2iree conversion pass targets simple cases and adheres closely to the official guidelines provided in the MLIR Dialect Conversion framework documentation:

https://mlir.llvm.org/docs/DialectConversion/


This is the current status of the implementations for the core rules and structure of the MLIR DialectConversion framework. 


| Feature                     | Status              | Matched to Docs |
| --------------------------- | ------------------- | --------------- |
| ConversionTarget setup      | ✅ Yes              | ✅ Fully matched |
| OpConversionPattern use     | ✅ Yes              | ✅ Fully matched |
| Remapped operands           | ✅ Used             | ✅ Fully matched |
| Conversion mode             | ✅ Partial          | ✅ Proper use    |
| TypeConverter               | ❌ Not needed (yet) | ✅ Optional      |
| Recursive legality          | ❌ Not needed (yet) | ✅ Optional      |
| Legal operation enforcement | ✅ Yes              | ✅ Fully matched |



## Requirements


### 🧰 Required Tools

To build and run this project, you'll need a few essential tools:

- **LLVM and MLIR**: These provide the core infrastructure for defining dialects, IR, and transformation passes.
- **CMake (version 3.13 or higher)**: Required as the build system.
- **Ninja** (optional): A faster alternative to traditional build systems like `make`.
- **Python 3**: Used to run the test harness (`run_test.py`).
- **IREE tools** (`iree-compile`, `iree-run-module`): Required to compile the transformed MLIR into executable formats and run them for validation.



## Build 

mkdir build 

cd build

cmake -G Ninja ..

ninga

