#include "mlir/IR/Dialect.h"
#include "llvm/Support/raw_ostream.h"

#include "Conversion/QuakeToStandard.h"
#include "Dialect/Quake/QuakeDialect.h"
#include "Dialect/CC/CCDialect.h"
#include "Dialect/CC/CCOps.h"
#include "Dialect/CC/CCTypes.h"


#include "mlir/InitAllPasses.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/IR/MLIRContext.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/SCF/IR/SCF.h"


using namespace mlir;

int main(int argc, char **argv) {
  DialectRegistry registry;

  // 1) Register *exactly* the dialects you need—including Quake—and any std ones
  registry.insert<
    arith::ArithDialect,
    func::FuncDialect,
    scf::SCFDialect,
    tensor::TensorDialect,
    complex::ComplexDialect,
    quake::QuakeDialect,
    cudaq::cc::CCDialect
  >();


  //registerAllDialects(registry);
  registerAllPasses();                    // Optional — only needed if you want standard MLIR passes
  quake::registerQuakeToStandardPass();  // Your custom pass

  // Register all the dialects with MLIRContext
  // Create the context with *that* registry
  /*
  MLIRContext context;
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();  


  // Now check: can we *load* the Quake dialect?
  if (auto *qd = context.getOrLoadDialect<quake::QuakeDialect>())
    llvm::errs() << "✅ quake::QuakeDialect is present!\n";
  else
    llvm::errs() << "❌ quake::QuakeDialect *NOT* present.\n";
  
  */
 

  return asMainReturnCode(
    MlirOptMain(argc, argv, "Quake→Standard converter\n", registry));
}
