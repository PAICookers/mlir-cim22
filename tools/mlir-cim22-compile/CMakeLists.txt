add_llvm_executable(mlir-cim22-compile mlir-cim22-compile.cpp)
llvm_update_compile_flags(mlir-cim22-compile)
target_link_libraries(mlir-cim22-compile PRIVATE MLIRCIM22ObjectEmitter)
mlir_check_all_link_libraries(mlir-cim22-compile)
