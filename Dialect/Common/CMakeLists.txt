add_mlir_library(QuakeCommon
  InlinerInterface.h
  Traits.h

  DEPENDS
    MLIRIR
    MLIRSupport

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRSupport
)
