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

add_library(MLIRCIM22RuntimeModel STATIC
  "${PROJECT_SOURCE_DIR}/runtime/src/ModelArtifact.cpp"
  ${_cim22_proto_cpp})
llvm_update_compile_flags(MLIRCIM22RuntimeModel)
target_include_directories(MLIRCIM22RuntimeModel PRIVATE
  "${PROJECT_SOURCE_DIR}/include"
  "${_cim22_generated}")
target_link_libraries(MLIRCIM22RuntimeModel PUBLIC
  MLIRCIM22Execution LLVMSupport protobuf::libprotobuf-lite)
