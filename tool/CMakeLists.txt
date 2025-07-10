include_directories(
  ${CMAKE_CURRENT_BINARY_DIR}
  ${PROJECT_BINARY_DIR}                # <--- ADD THIS
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}
)

add_executable(q2i-opt q2i-opt.cpp)

target_link_libraries(q2i-opt
  PRIVATE
  CCDialect     
  QuakeDialect         # <== Your dialect
  QuakeToStandard      # <== Your pass  
  MLIRSupport
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRDialect
  MLIRFuncDialect
  MLIRArithDialect
  #MLIRSCFDialect
  MLIRTensorDialect
  MLIRMlirOptMain
  
)
