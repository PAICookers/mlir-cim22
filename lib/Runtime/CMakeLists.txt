# Execution-only deployment library; LLVMSupport is linked statically.
add_library(cim22_runtime SHARED Runtime.cpp ModelData.cpp
  $<TARGET_OBJECTS:obj.MLIRCIM22Execution>
  $<TARGET_OBJECTS:obj.MLIRCIM22Support>)
llvm_update_compile_flags(cim22_runtime)
target_link_libraries(cim22_runtime PRIVATE LLVMSupport
  ${CMAKE_DL_LIBS})

# ExecutionEngine is excluded from libMLIR; its exported target carries the
# native codegen/JIT dependencies for either static or shared LLVM/MLIR builds.
add_library(cim22_compiler SHARED Compiler.cpp)
llvm_update_compile_flags(cim22_compiler)
target_link_libraries(cim22_compiler PRIVATE
  cim22_runtime MLIRCIM22Compiler MLIRExecutionEngine)
mlir_target_link_libraries(cim22_compiler PRIVATE
  MLIRBuiltinToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
  MLIRTargetLLVMIRExport)
set_target_properties(cim22_runtime cim22_compiler PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${LLVM_LIBRARY_OUTPUT_INTDIR}")
