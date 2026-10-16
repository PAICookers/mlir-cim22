set(_cim22_proto "${PROJECT_SOURCE_DIR}/schema/cim22_model.proto")
set(_cim22_generated "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${_cim22_generated}")
protobuf_generate(LANGUAGE cpp OUT_VAR _cim22_proto_cpp
  PROTOS "${_cim22_proto}" IMPORT_DIRS "${PROJECT_SOURCE_DIR}/schema"
  PROTOC_OUT_DIR "${_cim22_generated}" DEPENDENCIES protobuf::protoc)
protobuf_generate(LANGUAGE python OUT_VAR _cim22_proto_python
  PROTOS "${_cim22_proto}" IMPORT_DIRS "${PROJECT_SOURCE_DIR}/schema"
  PROTOC_OUT_DIR "${_cim22_generated}" DEPENDENCIES protobuf::protoc)
add_custom_target(check-cim22-protobuf
  COMMAND ${CMAKE_COMMAND} -E compare_files "${_cim22_proto_python}"
    "${PROJECT_SOURCE_DIR}/python/tests/_generated/cim22_model_pb2.py"
  DEPENDS ${_cim22_proto_python}
  COMMENT "Checking test Python Protobuf binding freshness" VERBATIM)
add_custom_target(update-cim22-protobuf
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_cim22_proto_python}"
    "${PROJECT_SOURCE_DIR}/python/tests/_generated/cim22_model_pb2.py"
  DEPENDS ${_cim22_proto_python}
  COMMENT "Regenerating test Python Protobuf binding" VERBATIM)

# Execution-only deployment library; Protobuf lite and LLVMSupport are static.
option(CIM22_ENABLE_USB "Build the Linux libusb runtime" ON)
if(CIM22_ENABLE_USB)
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "USB runtime requires Linux; set CIM22_ENABLE_USB=OFF")
  endif()
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(LIBUSB REQUIRED IMPORTED_TARGET GLOBAL libusb-1.0)
endif()

set(_cim22_runtime_sources Runtime.cpp ModelData.cpp CIMBoardRunner.cpp)
if(TARGET PkgConfig::LIBUSB)
  find_package(Threads REQUIRED)
  list(APPEND _cim22_runtime_sources UsbCIMDevice.cpp UsbRecordCodec.cpp
       LibusbUsbDriver.cpp)
endif()
add_library(cim22_runtime SHARED ${_cim22_runtime_sources}
  ${_cim22_proto_cpp}
  $<TARGET_OBJECTS:obj.MLIRCIM22Execution>
  $<TARGET_OBJECTS:obj.MLIRCIM22Support>)
llvm_update_compile_flags(cim22_runtime)
target_include_directories(cim22_runtime PRIVATE "${_cim22_generated}")
target_link_libraries(cim22_runtime PRIVATE LLVMSupport protobuf::libprotobuf-lite
  ${CMAKE_DL_LIBS})
if(TARGET PkgConfig::LIBUSB)
  target_compile_definitions(cim22_runtime PRIVATE CIM22_HAS_USB=1)
  target_link_libraries(cim22_runtime PRIVATE PkgConfig::LIBUSB Threads::Threads)
endif()

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

# Keep the compiler adapter's runtime dependency local to the wheel.
if(SKBUILD)
  set_target_properties(cim22_runtime cim22_compiler PROPERTIES
    INSTALL_RPATH "$ORIGIN")
  install(TARGETS cim22_runtime cim22_compiler
    LIBRARY DESTINATION lib
    COMPONENT runtime)
endif()
