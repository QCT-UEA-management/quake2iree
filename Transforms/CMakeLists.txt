
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

add_dependencies(QuakeToIREE QuakeDialect)
