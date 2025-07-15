#include "Conversion/QuakeToStandard.h"
#include "Dialect/Quake/QuakeDialect.h"
#include "Dialect/CC/CCDialect.h"

#include "mlir/InitAllPasses.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/IR/MLIRContext.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
//#include "mlir/Dialect/SCF/IR/SCF.h"


using namespace mlir;

int main(int argc, char **argv) {
  DialectRegistry registry;

  //  Register only the dialects you actually need
  registry.insert<
    //func::FuncDialect,
    //tensor::TensorDialect,
    quake::QuakeDialect,
    cudaq::cc::CCDialect
  >();

  registerAllDialects(registry);
  registerAllPasses();                    // Optional — only needed if you want standard MLIR passes
  quake::registerQuakeToStandardPass();  // Your custom pass


  // Register a CLI pipeline option like --quake-to-standard
  /*
  PassPipelineRegistration<> pipeline(
    "quake-to-standard",
    "Lower Quake dialect to standard dialects",
    quakeToStandardPipeline
  );
  */

  // Register all the dialects with MLIRContext
  MLIRContext context;
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();  // <-- this is critical

  

  return asMainReturnCode(
    MlirOptMain(argc, argv, "Quake optimizer\n", registry));
}
