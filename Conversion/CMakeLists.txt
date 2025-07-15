include_directories(
  ${CMAKE_CURRENT_BINARY_DIR}
  ${PROJECT_BINARY_DIR}                # <--- ADD THIS
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}
)

add_mlir_library(QuakeToStandard
  QuakeToStandard.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/lib/Conversion/

  DEPENDS
  QuakeToStandardPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  QuakeDialect
  CCDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRIR
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRTransforms
)

# TableGen: generate pass declarations
set(LLVM_TARGET_DEFINITIONS QuakeToStandard.td)
mlir_tablegen(QuakeToStandard.h.inc -gen-pass-decls -name QuakeToStandard)
add_dependencies(mlir-headers QuakeOpsIncGen)
add_public_tablegen_target(QuakeToStandardPassIncGen)

# Optional: documentation
add_mlir_doc(QuakeToStandard QuakeToStandard QuakeToStandard/ -gen-pass-doc)


