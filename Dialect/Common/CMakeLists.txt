add_mlir_library(QuakeCommon
  QuakeCommon.cpp
  InlinerInterface.h
  Traits.h
  Traits.td

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRSupport
)
