add_mlir_dialect_library(MLIRCIM22Dialect
  CIMDialect.cpp
  CIMOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/CIM22/Dialect/CIM/IR

  DEPENDS
  MLIRCIMOpsIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRSupport
)
