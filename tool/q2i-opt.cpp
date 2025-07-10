#include "Conversion/QuakeToStandard.h"
#include "Dialect/Quake/QuakeDialect.h"

#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/SCF/IR/SCF.h"


using namespace mlir;

int main(int argc, char **argv) {
  DialectRegistry registry;

  //  Register only the dialects you actually need
  registry.insert<
    arith::ArithDialect,
    func::FuncDialect,
    tensor::TensorDialect,
    quake::QuakeDialect
  >();

  //  Register core + custom passes
  //registerAllPasses();                    // Optional — only needed if you want standard MLIR passes
  quake::registerQuakeToStandardPass();  // Your custom pass

  return asMainReturnCode(
    MlirOptMain(argc, argv, "Quake optimizer\n", registry));
}
