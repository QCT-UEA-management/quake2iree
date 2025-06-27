set(LLVM_TARGET_DEFINITIONS QuakeOps.td)
mlir_tablegen(QuakeOps.h.inc -gen-op-decls)
mlir_tablegen(QuakeOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(CCQuakeOpsIncGen)

set(LLVM_TARGET_DEFINITIONS QuakeTypes.td)
mlir_tablegen(QuakeTypes.h.inc -gen-typedef-decls)
mlir_tablegen(QuakeTypes.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(CCQuakeTypesIncGen)

set(LLVM_TARGET_DEFINITIONS QuakeInterfaces.td)
mlir_tablegen(QuakeInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(QuakeInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(CCQuakeInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS QuakeDialect.td)
mlir_tablegen(QuakeDialect.h.inc -gen-dialect-decls)
mlir_tablegen(QuakeDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(CCQuakeDialectIncGen)

add_mlir_dialect_library(CCDialect
  QuakeOps.cpp
  QuakeTypes.cpp
  QuakeInterfaces.cpp
  QuakeDialect.cpp

  DEPENDS
    CCQuakeOpsIncGen
    CCQuakeTypesIncGen
    CCQuakeInterfacesIncGen
    CCQuakeDialectIncGen

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRSupport
)
