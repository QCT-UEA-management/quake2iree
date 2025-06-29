include_directories(
  ${CMAKE_CURRENT_BINARY_DIR}
  ${PROJECT_BINARY_DIR}                # <--- ADD THIS
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}
)


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

# Ensure proper dependencies for generated headers/macros
add_dependencies(QuakeToIREE
  QuakeDialect
)

mlir_check_all_link_libraries(quake2iree-opt)
