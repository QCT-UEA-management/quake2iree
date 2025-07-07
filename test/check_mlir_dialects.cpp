#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"

#include <iostream>

int main() {
  mlir::DialectRegistry registry;


  registry.insert<mlir::func::FuncDialect,
                  mlir::scf::SCFDialect,
                  mlir::AffineDialect>();

  mlir::MLIRContext context;
  context.appendDialectRegistry(registry);

  context.loadAllAvailableDialects();

  std::cout << "MLIR context initialized and core dialects loaded.\n";
  return 0;
}
