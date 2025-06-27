# Inherit paths from the root CMake
include(AddMLIR)
include(AddLLVM)

add_executable(check_mlir_dialects check_mlir_dialects.cpp)

# Optional but useful — only if defined
if(COMMAND llvm_update_compile_flags)
  llvm_update_compile_flags(check_mlir_dialects)
endif()

target_link_libraries(check_mlir_dialects
  PRIVATE
  MLIRFuncDialect
  MLIRSCFDialect
  MLIRAffineDialect
  MLIRIR
  MLIRSupport
  LLVMSupport
)

mlir_check_all_link_libraries(check_mlir_dialects)
