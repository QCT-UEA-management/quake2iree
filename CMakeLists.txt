# Project name and language specification
cmake_minimum_required(VERSION 3.19...3.30)
project(
  MLIRPasses
  VERSION 1.0
  LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED on)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
set(CMAKE_CXX_FLAGS "-w -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-switch")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frtti -fexceptions")

# Set default build type to Debug if none is specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      Debug
      CACHE STRING "Build type (Debug, Release)" FORCE)
endif()

# Define macros for Debug builds
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  message("Configuring for Debug mode")
  add_compile_definitions(DEBUG)
endif()

# ################ MLIR Settings ###########################################
# Find MLIR. (MLIR is usually installed as part of an LLVM distribution.)
if(NOT MLIR_DIR AND EXISTS "/usr/local/llvm/lib/cmake/mlir")
  set(MLIR_DIR "/usr/local/llvm/lib/cmake/mlir")
endif()

find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

# Append MLIR's CMake modules.
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")

# Include the MLIR helper macros.
include(TableGen)
include(AddMLIR)
include(AddLLVM)
# Add LLVM, Clang and MLIR headers to the include path
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})


# Add your dialect, transforms, and tests
add_subdirectory(Dialect)
add_subdirectory(Conversion)
add_subdirectory(tool)

#add_subdirectory(test)

mlir_check_all_link_libraries(quake_to_standard-opt)