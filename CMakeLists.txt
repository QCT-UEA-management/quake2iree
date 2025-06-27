cmake_minimum_required(VERSION 3.13)
project(CheckMLIRDialects)

set(CMAKE_CXX_STANDARD 17)

# Set your LLVM and MLIR install directory here
set(LLVM_DIR "/usr/local/opt/llvm/lib/cmake/llvm")
set(MLIR_DIR "/usr/local/opt/llvm/lib/cmake/mlir")

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found MLIR: ${MLIR_DIR}")
message(STATUS "Found LLVM: ${LLVM_PACKAGE_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(AddMLIR)
include(AddLLVM)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(test)
