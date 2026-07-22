add_mlir_dialect_library(MLIRCIM22CIMFrameDialect
  CIMFrameDialect.cpp
  CIMFrameOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/CIM22/Dialect/CIMFrame/IR

  DEPENDS
  MLIRCIMFrameOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
