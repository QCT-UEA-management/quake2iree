
include_directories(
  ${CMAKE_CURRENT_BINARY_DIR}
  ${PROJECT_BINARY_DIR}                # <--- ADD THIS
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}
)

set(DIALECT_TD_INCLUDE_DIRS
  -I ${PROJECT_SOURCE_DIR}  # ← this is the key path!
  -I ${CMAKE_CURRENT_SOURCE_DIR}
  -I ${CMAKE_CURRENT_SOURCE_DIR}/..
  -I ${CMAKE_CURRENT_SOURCE_DIR}/../Common
  -I ${CMAKE_CURRENT_SOURCE_DIR}/../CC
)

set(LLVM_TARGET_DEFINITIONS QuakeOps.td)
mlir_tablegen(QuakeOps.h.inc -gen-op-decls ${DIALECT_TD_INCLUDE_DIRS})
mlir_tablegen(QuakeOps.cpp.inc -gen-op-defs ${DIALECT_TD_INCLUDE_DIRS})
add_public_tablegen_target(QuakeOpsIncGen)

set(LLVM_TARGET_DEFINITIONS QuakeTypes.td)
mlir_tablegen(QuakeTypes.h.inc -gen-typedef-decls -typedefs-dialect=quake ${DIALECT_TD_INCLUDE_DIRS})
mlir_tablegen(QuakeTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=quake ${DIALECT_TD_INCLUDE_DIRS})
add_public_tablegen_target(QuakeTypesIncGen)

set(LLVM_TARGET_DEFINITIONS QuakeInterfaces.td)
mlir_tablegen(QuakeInterfaces.h.inc -gen-op-interface-decls ${DIALECT_TD_INCLUDE_DIRS})
mlir_tablegen(QuakeInterfaces.cpp.inc -gen-op-interface-defs ${DIALECT_TD_INCLUDE_DIRS})
add_public_tablegen_target(QuakeInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS QuakeDialect.td)
mlir_tablegen(QuakeDialect.h.inc -gen-dialect-decls ${DIALECT_TD_INCLUDE_DIRS})
mlir_tablegen(QuakeDialect.cpp.inc -gen-dialect-defs ${DIALECT_TD_INCLUDE_DIRS})
add_public_tablegen_target(QuakeDialectIncGen)


# Header-only MLIR dialect library
add_mlir_dialect_library(QuakeDialect
  QuakeDialect.cpp 
  QuakeInterfaces.cpp 
  QuakeOps.cpp
  QuakeTypes.cpp
  Factory.cpp
  Intrinsics.cpp

  DEPENDS
    QuakeOpsIncGen
    QuakeTypesIncGen
    QuakeInterfacesIncGen
    QuakeDialectIncGen

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRSupport
)