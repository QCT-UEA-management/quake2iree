find_package(MLIR REQUIRED CONFIG)

add_mlir_dialect_library(MyConversion
  MyPass.cpp

  LINK_LIBS PUBLIC
    MLIRArithDialect
    MLIRTensorDialect
    MLIRFuncDialect
    MLIRTransforms
    MLIRIR
)
