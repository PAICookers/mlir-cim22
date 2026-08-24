add_mlir_library(MLIRCIM22Execution
  CIMSoftwareRunner.cpp
  CIMUartRunner.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/CIM22/Execution

  LINK_LIBS PUBLIC
  LLVMSupport
)
