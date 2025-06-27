# quake2iree

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


## Build 

mkdir build 

cd build

cmake -G Ninja ..

ninga
