include_directories(  
  ${CMAKE_CURRENT_BINARY_DIR}  
  ${PROJECT_BINARY_DIR}  
  ${CMAKE_CURRENT_SOURCE_DIR}  
  ${PROJECT_SOURCE_DIR}  
)  
  
set(CONVERSION_INCLUDE_DIRS  
  -I ${PROJECT_SOURCE_DIR}  
  -I ${CMAKE_CURRENT_SOURCE_DIR}  
  -I ${CMAKE_CURRENT_SOURCE_DIR}/..
  -I ${CMAKE_CURRENT_SOURCE_DIR}/../Quake  
)  
  
add_mlir_library(QuakeToStandard  
  QuakeToStandard.cpp  
  
  ADDITIONAL_HEADER_DIRS  
    ${PROJECT_SOURCE_DIR}/Conversion/  
  
  DEPENDS  
    QuakeToStandardPassIncGen  
  
  LINK_COMPONENTS  
    Core  
  
  LINK_LIBS PUBLIC  
    QuakeDialect  
    CCDialect  
    MLIRArithDialect
    MLIRFuncDialect
    MLIRFuncTransforms
    MLIRIR
    MLIRLinalgDialect
    MLIRMemRefDialect
    MLIRMathDialect
    MLIRPass
    MLIRSCFDialect
    MLIRTensorDialect
    MLIRTransforms
)  

set(LLVM_TARGET_DEFINITIONS QuakeToStandard.td)  
mlir_tablegen(QuakeToStandard.h.inc -gen-pass-decls -name QuakeToStandard ${CONVERSION_INCLUDE_DIRS}) 
add_dependencies(mlir-headers QuakeOpsIncGen)
add_public_tablegen_target(QuakeToStandardPassIncGen)  
add_mlir_doc(QuakeToStandard QuakeToStandard QuakeToStandard/ -gen-pass-doc)
