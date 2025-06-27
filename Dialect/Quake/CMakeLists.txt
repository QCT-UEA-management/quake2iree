set(LLVM_TARGET_DEFINITIONS QuakeOps.td)
mlir_tablegen(QuakeOps.h.inc -gen-op-decls)
mlir_tablegen(QuakeOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(QuakeOpsIncGen)

set(LLVM_TARGET_DEFINITIONS QuakeTypes.td)
mlir_tablegen(QuakeTypes.h.inc -gen-typedef-decls)
mlir_tablegen(QuakeTypes.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(QuakeTypesIncGen)

set(LLVM_TARGET_DEFINITIONS QuakeInterfaces.td)
mlir_tablegen(QuakeInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(QuakeInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(QuakeInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS QuakeDialect.td)
mlir_tablegen(QuakeDialect.h.inc -gen-dialect-decls)
mlir_tablegen(QuakeDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(QuakeDialectIncGen)

# Header-only MLIR dialect library
add_mlir_dialect_library(QuakeDialect

  # You don't have any .cpp sources, just public headers
  QuakeDialect.cpp  # dummy .cpp files will make CMake happy
  QuakeDialect.h
  QuakeOps.h
  QuakeTypes.h
  QuakeInterfaces.h

  DEPENDS
    QuakeOpsIncGen
    QuakeTypesIncGen
    QuakeInterfacesIncGen
    QuakeDialectIncGen

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRSupport
)
