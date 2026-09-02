add_mlir_library(MLIRCIM22Target
  CIMFrameCodec.cpp
  CIMExecutable.cpp
  CIMRTLArtifactExporter.cpp
  Passes.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/CIM22/Target

  DEPENDS
  MLIRCIM22TargetPassesIncGen

  LINK_LIBS PUBLIC
  MLIRCIM22Dialect
  MLIRCIM22CIMFrameDialect
  MLIRCIM22LinalgToCIM
  MLIRCIM22Support
  MLIRCIM22Execution
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  LLVMSupport
)
