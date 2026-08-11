add_llvm_executable(mlir-cim22-onnx-import mlir-cim22-onnx-import.cpp)

llvm_update_compile_flags(mlir-cim22-onnx-import)
target_link_libraries(mlir-cim22-onnx-import PRIVATE
  MLIRCIM22ONNXFrontend
)

mlir_check_all_link_libraries(mlir-cim22-onnx-import)
