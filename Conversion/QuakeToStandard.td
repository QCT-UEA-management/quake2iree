#ifndef CONVERSION_QUAKETOSTANDARD_TD_
#define CONVERSION_QUAKETOSTANDARD_TD_

include "mlir/Pass/PassBase.td"

def QuakeToStandard : Pass<"quake-to-standard", "mlir::ModuleOp"> {
  let summary = "Lower `quake` to standard MLIR dialects.";

  let description = [{
    This pass lowers the `quake` dialect to standard MLIR, a mixture of affine,
    tensor, and arith.
  }];

  //let constructor = "::mlir::quake_to_standard::QuakeToStandard";

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::scf::SCFDialect",
    "mlir::tensor::TensorDialect",
    "quake::QuakeDialect",
    "cudaq::cc::CCDialect",
    "mlir::LLVM::LLVMDialect",
  ];
}

#endif  // CONVERSION_QUAKETOSTANDARD_TD_
