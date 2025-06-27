#ifndef QUAKE_TRANSFORMS_PASSES_H
#define QUAKE_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"


// This is a header file to declare and register the pass.

namespace mlir {
namespace quake {

/// Creates a pass to convert Quake dialect ops to standard MLIR dialects.
std::unique_ptr<mlir::Pass> createConvertQuakeToIREEPass();

/// Registers all passes in the Quake to IREE pipeline.
void registerQuakeToIREEPass();

} // namespace quake
} // namespace mlir

#endif // QUAKE_TRANSFORMS_PASSES_H
