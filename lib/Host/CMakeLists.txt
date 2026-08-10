add_mlir_library(MLIRCIM22Host
  HostProgram.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/CIM22/Host

  LINK_LIBS PUBLIC
  MLIRCIM22Dialect
  MLIRFuncDialect
  MLIRIR
  MLIRSideEffectInterfaces
)
