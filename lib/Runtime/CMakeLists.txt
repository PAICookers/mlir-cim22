# Execution-only deployment library; LLVMSupport is linked statically.
add_library(cim22_runtime SHARED Runtime.cpp ModelData.cpp
  $<TARGET_OBJECTS:obj.MLIRCIM22Execution>
  $<TARGET_OBJECTS:obj.MLIRCIM22Support>)
llvm_update_compile_flags(cim22_runtime)
target_link_libraries(cim22_runtime PRIVATE LLVMSupport
  ${CMAKE_DL_LIBS})

# The packaged ExecutionEngine links the monolithic MLIR dylib. Embedding our
# object libraries avoids pulling a second copy through static link interfaces.
add_library(cim22_compiler SHARED Compiler.cpp
  $<TARGET_OBJECTS:obj.MLIRCIM22Compiler>
  $<TARGET_OBJECTS:obj.MLIRCIM22LinalgToCIM>
  $<TARGET_OBJECTS:obj.MLIRCIM22Dialect>
  $<TARGET_OBJECTS:obj.MLIRCIM22CIMFrameDialect>
  $<TARGET_OBJECTS:obj.MLIRCIM22CIMFrameTransforms>
  $<TARGET_OBJECTS:obj.MLIRCIM22Target>)
llvm_update_compile_flags(cim22_compiler)
target_link_libraries(cim22_compiler PRIVATE cim22_runtime MLIR LLVM)
set_target_properties(cim22_runtime cim22_compiler PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${LLVM_LIBRARY_OUTPUT_INTDIR}")
