add_mlir_library(MLIRCIM22RegisterAll
  PARTIAL_SOURCES_INTENDED
  RegisterAll.cpp
  Pipelines.cpp
  LINK_LIBS PUBLIC
  MLIRCIM22LinalgToCIM
  MLIRCIM22CIMFrameTransforms
  MLIRCIM22Target
  MLIRArithDialect
  MLIRArithTransforms
  MLIRBufferizationDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRLinalgDialect
  MLIRLLVMDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRRegisterAllPasses
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRTensorDialect
  MLIRTensorInferTypeOpInterfaceImpl
  MLIRTensorTransforms
)

add_mlir_library(MLIRCIM22ObjectEmitter
  PARTIAL_SOURCES_INTENDED
  ObjectEmitter.cpp
  LINK_LIBS PUBLIC
  MLIRCIM22RegisterAll
  MLIRCIM22RuntimeModel
  MLIRExecutionEngine
)
mlir_target_link_libraries(MLIRCIM22ObjectEmitter PRIVATE
  MLIRBuiltinToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
  MLIRTargetLLVMIRExport)
