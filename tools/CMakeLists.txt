add_executable(quake2iree-opt
  quake2iree-opt.cpp
)

target_link_libraries(quake2iree-opt
  PRIVATE
    QuakeDialect
    QuakeToIREE
    MLIRArithDialect
    MLIRFuncDialect
    MLIRTensorDialect
    MLIRParser
    MLIRIR
    MLIRPass
    MLIRSupport
)

mlir_check_all_link_libraries(quake2iree-opt)
