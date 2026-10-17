add_subdirectory(mlir-cim22-opt)
install(TARGETS mlir-cim22-opt
  RUNTIME DESTINATION bin COMPONENT runtime)

if(MLIRCIM22_ENABLE_ONNX)
  add_subdirectory(mlir-cim22-onnx-import)
  install(TARGETS mlir-cim22-onnx-import
    RUNTIME DESTINATION bin COMPONENT dev)
endif()
