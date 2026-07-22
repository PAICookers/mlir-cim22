add_llvm_executable(mlir-cim22-opt mlir-cim22-opt.cpp)

llvm_update_compile_flags(mlir-cim22-opt)
target_link_libraries(mlir-cim22-opt PRIVATE
  MLIRCIM22Dialect
  MLIRCIM22CIMFrameDialect
  MLIRCIM22CIMFrameTransforms
  MLIRFuncDialect
  MLIROptLib
)

mlir_check_all_link_libraries(mlir-cim22-opt)
