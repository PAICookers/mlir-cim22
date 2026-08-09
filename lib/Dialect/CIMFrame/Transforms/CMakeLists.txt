add_mlir_library(MLIRCIM22CIMFrameTransforms
  Passes.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/CIM22/Dialect/CIMFrame/Transforms

  DEPENDS
  MLIRCIM22CIMFramePassesIncGen

  LINK_LIBS PUBLIC
  MLIRCIM22Dialect
  MLIRCIM22CIMFrameDialect
  MLIRCIM22Support
  MLIRIR
  MLIRPass
  MLIRTransforms
)
