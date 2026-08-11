add_mlir_library(MLIRCIM22Target
  CIMFrameCodec.cpp
  Passes.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/CIM22/Target

  DEPENDS
  MLIRCIM22TargetPassesIncGen

  LINK_LIBS PUBLIC
  MLIRCIM22Dialect
  MLIRCIM22CIMFrameDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
)
