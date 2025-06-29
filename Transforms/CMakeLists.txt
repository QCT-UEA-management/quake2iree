
include_directories(
  ${CMAKE_CURRENT_BINARY_DIR}
  ${PROJECT_BINARY_DIR}                # <--- ADD THIS
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}
)


add_mlir_library(QuakeToIREE
  QuakeToIree.cpp

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransforms
    MLIRArithDialect
    MLIRFuncDialect
    MLIRTensorDialect
    QuakeDialect
)


# Ensure proper dependencies for generated headers/macros
add_dependencies(QuakeToIREE
  QuakeDialect
)