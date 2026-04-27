include_directories(
  ${CMAKE_CURRENT_BINARY_DIR}
  ${PROJECT_BINARY_DIR}                # <--- ADD THIS
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}
)


set (LIBS
  CCDialect
  QuakeDialect         # <== Your dialect
  QuakeToStandard      # <== Your pass
  QuakeCommon
  MLIRSupport
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRDialect
  MLIRFuncDialect
  MLIRArithDialect
  MLIRTensorDialect
  MLIRLLVMDialect
  MLIRMlirOptMain
  MLIRIR
)


add_llvm_executable(q2i-opt q2i-opt.cpp)

llvm_update_compile_flags(q2i-opt)

target_link_libraries(q2i-opt PRIVATE ${LIBS})

target_sources(q2i-opt PRIVATE
  ${PROJECT_SOURCE_DIR}/Dialect/Quake/QuakeDialect.cpp
)

mlir_check_all_link_libraries(q2i-opt)


