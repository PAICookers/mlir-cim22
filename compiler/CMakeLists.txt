# Native object emission adapter. The MLIR registration aggregate remains in
# lib/Compiler; this shared library owns the ExecutionEngine boundary.
add_library(cim22_compiler SHARED NativeCompiler.cpp)
llvm_update_compile_flags(cim22_compiler)
target_include_directories(cim22_compiler PRIVATE
  "${PROJECT_SOURCE_DIR}/runtime")
target_link_libraries(cim22_compiler PRIVATE
  cim22_runtime MLIRCIM22RegisterAll MLIRExecutionEngine)
mlir_target_link_libraries(cim22_compiler PRIVATE
  MLIRBuiltinToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
  MLIRTargetLLVMIRExport)
set_target_properties(cim22_compiler PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${LLVM_LIBRARY_OUTPUT_INTDIR}"
  LIBRARY_OUTPUT_DIRECTORY "${LLVM_LIBRARY_OUTPUT_INTDIR}"
  RUNTIME_OUTPUT_DIRECTORY "${LLVM_LIBRARY_OUTPUT_INTDIR}"
  WINDOWS_EXPORT_ALL_SYMBOLS ON)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_options(cim22_compiler PRIVATE
    "LINKER:--version-script=${PROJECT_SOURCE_DIR}/runtime/cim22.exports")
  set_property(TARGET cim22_compiler APPEND PROPERTY LINK_DEPENDS
    "${PROJECT_SOURCE_DIR}/runtime/cim22.exports")
endif()

if(SKBUILD)
  set_target_properties(cim22_compiler PROPERTIES
    INSTALL_RPATH "$ORIGIN")
  install(TARGETS cim22_compiler
    LIBRARY DESTINATION lib COMPONENT runtime
    RUNTIME DESTINATION lib COMPONENT runtime
    ARCHIVE DESTINATION lib COMPONENT runtime)
endif()
