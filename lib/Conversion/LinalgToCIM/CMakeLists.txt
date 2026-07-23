add_mlir_library(MLIRCIM22LinalgToCIM
  Passes.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/CIM22/Conversion/LinalgToCIM

  DEPENDS
  MLIRCIM22LinalgToCIMPassesIncGen

  LINK_LIBS PUBLIC
  MLIRCIM22Dialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
)
