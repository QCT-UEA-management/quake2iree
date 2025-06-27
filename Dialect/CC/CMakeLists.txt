include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${CMAKE_CURRENT_SOURCE_DIR}/../Common
  ${CMAKE_CURRENT_SOURCE_DIR}/../CC
)

set(LLVM_TARGET_DEFINITIONS CCOps.td)
mlir_tablegen(CCOps.h.inc -gen-op-decls)
mlir_tablegen(CCOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(CCOpsIncGen)

set(LLVM_TARGET_DEFINITIONS CCTypes.td)
mlir_tablegen(CCTypes.h.inc -gen-typedef-decls)
mlir_tablegen(CCTypes.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(CCTypesIncGen)

set(LLVM_TARGET_DEFINITIONS CCInterfaces.td)
mlir_tablegen(CCInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(CCInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(CCInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS CCDialect.td)
mlir_tablegen(CCDialect.h.inc -gen-dialect-decls)
mlir_tablegen(CCDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(CCDialectIncGen)

# Dummy source file (you need to create this)
add_mlir_dialect_library(CCDialect
  CCDialect.cpp  # <-- create this minimal file

  CCDialect.h
  CCOps.h
  CCTypes.h
  CCInterfaces.h

  DEPENDS
    CCOpsIncGen
    CCTypesIncGen
    CCInterfacesIncGen
    CCDialectIncGen

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRSupport
)
