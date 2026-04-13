# Writing an MLIR Dialect Translator (Dialect Conversion)
In MLIR, translating one dialect to others is called **dialect conversion** or **lowering**. Here's the structured path to implement it:

## 1. Understand Your Source & Target Dialects
Before writing any code, answer:

- What ops does your **source dialect** have?
- Which **target dialects** will you lower to? (e.g., arith, func, memref, llvm, scf)
- Is this a **full lowering** (no source ops remain) or **partial lowering**?

## 2. Set Up Your Pass Infrastructure
Create a **conversion pass** using MLIR's pass framework:

```c++
// MyDialectToTargetPass.h
#include "mlir/Pass/Pass.h"

std::unique_ptr<mlir::Pass> createMyDialectToTargetPass();
```

Register it in your `.td` file:

```tablegen
// Passes.td
def MyDialectToTarget : Pass<"convert-my-dialect-to-target", "mlir::ModuleOp"> {
  let summary = "Convert MyDialect to Target dialects";
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::func::FuncDialect"];
}
```

## 3. Define a Type Converter (if needed)
If your source dialect has **custom types**, map them to target types:

```c++
class MyTypeConverter : public mlir::TypeConverter {
public:
  MyTypeConverter() {
    // Identity conversion for standard types
    addConversion([](mlir::Type type) { return type; });

    // Convert MyType → MemRefType
    addConversion([](MyType type) -> mlir::Type {
      return mlir::MemRefType::get({type.getSize()}, type.getElementType());
    });
  }
};
```

## 4. Write Conversion Patterns (the core work)
Each pattern handles **one op** from your source dialect:

```c++
struct MyOpConversionPattern : public mlir::OpConversionPattern<my_dialect::MyOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      my_dialect::MyOp op,
      OpAdaptor adaptor,           // operands already type-converted
      mlir::ConversionPatternRewriter &rewriter) const override {

    // Replace source op with target ops
    rewriter.replaceOpWithNewOp<mlir::arith::AddIOp>(
        op, adaptor.getLhs(), adaptor.getRhs());

    return mlir::success();
  }
};
```

Key rules:

- Use `adaptor` (not `op`) to access **already-converted operands**
- Use `rewriter.replaceOpWithNewOp<>` or `rewriter.replaceOp()`
- Return `failure()` if the pattern cannot apply

## 5. Populate Patterns & Configure the Conversion Target
```c++
void populateMyDialectToTargetPatterns(
    mlir::TypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns) {

  patterns.add
    MyOpConversionPattern,
    MyOtherOpConversionPattern
  >(typeConverter, patterns.getContext());
}
```

Then set up the **ConversionTarget** (what is legal after conversion):

```c++
mlir::ConversionTarget target(getContext());

// Target dialects are legal
target.addLegalDialect<mlir::arith::ArithDialect>();
target.addLegalDialect<mlir::func::FuncDialect>();

// Source dialect ops are illegal (must be converted)
target.addIllegalDialect<my_dialect::MyDialect>();

// Or mark specific ops illegal/legal with predicates:
target.addDynamicallyLegalOp<mlir::func::FuncOp>([&](mlir::func::FuncOp op) {
  return typeConverter.isSignatureLegal(op.getFunctionType());
});
```

## 6. Run the Conversion in Your Pass
```c++
void runOnOperation() override {
  mlir::ModuleOp module = getOperation();
  MyTypeConverter typeConverter;

  mlir::RewritePatternSet patterns(&getContext());
  populateMyDialectToTargetPatterns(typeConverter, patterns);

  // Also add function signature conversion if needed
  mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(
      patterns, typeConverter);

  if (mlir::failed(mlir::applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}
```

Choose the right conversion driver:

| **Function**              | **Use when**                           |
| ------------------------- | -------------------------------------- | 
| `applyFullConversion`     | All source ops **must** be converted   |
| `applyPartialConversion`  | Some source ops may **legally remain** |
| `applyAnalysisConversion` | Dry-run to check feasibility           |

## 7. Handle Edge Cases

- **Regions & blocks**: Use `ConversionPatternRewriter::convertRegionTypes()` for ops with regions (e.g., loops, functions)
- **1-to-N type mappings**: Use `OneToNTypeConverter` for types that expand into multiple values
- **Op with multiple results**: Handle each result mapping explicitly
- **Unrealized casts**: MLIR may insert `UnrealizedConversionCastOp` as bridges — clean them up with `reconcileUnrealizedCasts()`

## 8. Test Your Lowering
Use `mlir-opt` + FileCheck:

```mlir
// test/my-lowering.mlir
// RUN: mlir-opt %s --convert-my-dialect-to-target | FileCheck %s

func.func @test(%a: i32, %b: i32) -> i32 {
  %0 = my_dialect.add %a, %b : i32
  return %0 : i32
}

// CHECK-LABEL: func.func @test
// CHECK: arith.addi
// CHECK-NOT: my_dialect.add
```

## Summary Flow

```
Source Dialect Op
      │
      ▼
ConversionPattern::matchAndRewrite()
      │  uses adaptor (converted operands)
      ▼
RewritePatternRewriter::replaceOpWithNewOp<TargetOp>()
      │
      ▼
ConversionTarget validates legality
      │
      ▼
applyFullConversion / applyPartialConversion
```

## Recommended References

- `mlir/lib/Transforms/Utils/DialectConversion.cpp` — the engine itself
- The `affine-to-standard`, `scf-to-cf`, and `arith-to-llvm` lowerings in the MLIR repo are excellent real-world examples to study
- MLIR docs: **"Dialect Conversion"** page on mlir.llvm.org