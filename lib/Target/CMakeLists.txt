add_mlir_library(MLIRCIM22Target
  Passes.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/CIM22/Target

  DEPENDS
  MLIRCIM22TargetPassesIncGen

  LINK_LIBS PUBLIC
  MLIRCIM22Dialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
)
