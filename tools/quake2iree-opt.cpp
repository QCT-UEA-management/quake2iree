// tools/quake2iree-opt.cpp
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "Dialect/Quake/QuakeDialect.h"
#include "Transforms/Passes.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  // Register MLIR core dialects
  registry.insert<mlir::arith::ArithDialect,
                  mlir::func::FuncDialect,
                  mlir::tensor::TensorDialect>();

  // Register your custom dialect
  registry.insert<mlir::quake::QuakeDialect>();

  // Register core and custom passes
  mlir::registerAllPasses();
  mlir::quake::registerQuakeToIREEPass();  // You must define this!

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Quake2IREE Pass Driver\n", registry));
}
