cmake_minimum_required(VERSION 3.13)
project(CheckMLIRDialects)

set(CMAKE_CXX_STANDARD 17)

# Set your LLVM and MLIR install directory here
set(LLVM_DIR "/usr/local/opt/llvm/lib/cmake/llvm")
set(MLIR_DIR "/usr/local/opt/llvm/lib/cmake/mlir")

# Ensure CMake can find LLVM and MLIR modules
list(APPEND CMAKE_MODULE_PATH "${MLIR_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_DIR}")

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found MLIR: ${MLIR_DIR}")
message(STATUS "Found LLVM: ${LLVM_PACKAGE_DIR}")

include(AddMLIR)
include(AddLLVM)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})
add_definitions(${LLVM_DEFINITIONS})

# ----------------------------------------------
# Add your dialect, transforms, and tests
add_subdirectory(Dialect)
add_subdirectory(Transforms)
add_subdirectory(test)
add_subdirectory(tools)

# ----------------------------------------------
# Build the main tool quake2iree-opt
add_executable(quake2iree-opt tools/quake2iree-opt.cpp)

target_link_libraries(quake2iree-opt
  PRIVATE
    QuakeToIREE
    QuakeDialect
    MLIRArithDialect
    MLIRFuncDialect
    MLIRTensorDialect
    MLIRIR
    MLIRPass
    MLIRParser
    MLIRSupport
)

mlir_check_all_link_libraries(quake2iree-opt)
