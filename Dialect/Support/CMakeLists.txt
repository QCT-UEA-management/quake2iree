
include_directories(
  ${CMAKE_CURRENT_BINARY_DIR}
  ${PROJECT_BINARY_DIR}                # <--- ADD THIS
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}
)

add_mlir_library(AuxSupport
  SmallVector.cpp
  SmallVector.h
  

  LINK_LIBS PUBLIC
    QuakeDialect
    CCDialect
)
