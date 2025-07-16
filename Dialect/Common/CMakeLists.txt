
include_directories(
  ${CMAKE_CURRENT_BINARY_DIR}
  ${PROJECT_BINARY_DIR}                
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}
)


# Generate TableGen files for Traits  
set(LLVM_TARGET_DEFINITIONS Traits.td)  
mlir_tablegen(Traits.h.inc --gen-attrdef-decls)  
mlir_tablegen(Traits.cpp.inc --gen-attrdef-defs)  
add_public_tablegen_target(CommonTraitsIncGen) 

add_mlir_library(QuakeCommon  
  QuakeCommon.cpp  
    
  ADDITIONAL_HEADER_DIRS  
    ${PROJECT_SOURCE_DIR}/Dialect/Common  
  
  DEPENDS  
    CommonTraitsIncGen
  
  LINK_LIBS PUBLIC  
    MLIRIR  
    MLIRSupport  
)
