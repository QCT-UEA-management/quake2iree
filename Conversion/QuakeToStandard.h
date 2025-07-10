#ifndef CONVERSION_QUAKETOSTANDARD_H_
#define CONVERSION_QUAKETOSTANDARD_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Arith/IR/Arith.h"   
#include "mlir/Dialect/Tensor/IR/Tensor.h"  

namespace mlir {
namespace quake_to_standard {

#define GEN_PASS_DECL
#include "Conversion/QuakeToStandard.h.inc"

#define GEN_PASS_REGISTRATION
#include "Conversion/QuakeToStandard.h.inc"

}  // namespace quake_to_standard
}  // namespace mlir


// ✅ Add this declaration so q2i-opt.cpp can call it
namespace quake {
  void registerQuakeToStandardPass();
}

#endif  // CONVERSION_QUAKETOSTANDARD_H_
