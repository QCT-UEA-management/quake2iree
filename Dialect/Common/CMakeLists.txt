
include_directories(
  ${CMAKE_CURRENT_BINARY_DIR}
  ${PROJECT_BINARY_DIR}                # <--- ADD THIS
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}
)

add_mlir_library(QuakeCommon
  QuakeCommon.cpp
  InlinerInterface.h
  Traits.h
  Traits.td

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRSupport
)
