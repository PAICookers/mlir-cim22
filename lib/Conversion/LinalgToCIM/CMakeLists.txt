add_mlir_library(MLIRCIM22LinalgToCIM
  CIMSegments.cpp
  Passes.cpp
  ExecutionPlanVerifier.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/CIM22/Conversion/LinalgToCIM

  DEPENDS
  MLIRCIM22LinalgToCIMPassesIncGen

  LINK_LIBS PUBLIC
  MLIRCIM22Dialect
  MLIRAnalysis
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRMemRefDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
)
