add_mlir_library(MLIRCIM22ONNXFrontend
  Importer.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/CIM22/Frontend/ONNX

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRSupport
  MLIRTensorDialect
)

# ONNX exports public compile definitions as well as headers and libraries.
# Use the MLIR helper so those requirements reach both the object and archive
# targets created by add_mlir_library.
mlir_target_link_libraries(MLIRCIM22ONNXFrontend PUBLIC
  MLIRCIM22ONNXProto
)
