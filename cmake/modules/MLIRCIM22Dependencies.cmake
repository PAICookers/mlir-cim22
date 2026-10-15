#===- MLIRCIM22Dependencies.cmake - Third-party dependencies --------------===#
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===#

include_guard(GLOBAL)

include(FetchContent)

option(MLIRCIM22_USE_SYSTEM_ONNX
  "Use an installed ONNX package instead of the pinned source dependency"
  OFF
)

if(MLIRCIM22_USE_SYSTEM_ONNX)
  find_package(ONNX 1.21.0 EXACT CONFIG REQUIRED)
  # Use the same package as ONNX; never fetch a second Protobuf in system mode.
  find_package(Protobuf 31.1.0 EXACT CONFIG QUIET)
  if(NOT Protobuf_FOUND OR NOT TARGET protobuf::protoc OR
      NOT TARGET protobuf::libprotobuf-lite)
    message(FATAL_ERROR
      "System ONNX requires matching Protobuf 31.1.0 CONFIG, protoc and "
      "static protobuf::libprotobuf-lite targets for CIM22 artifacts")
  endif()
  get_target_property(_cim22_protoc protobuf::protoc IMPORTED_LOCATION)
  if(NOT _cim22_protoc)
    get_target_property(_cim22_protoc_configs protobuf::protoc IMPORTED_CONFIGURATIONS)
    foreach(_cim22_config IN LISTS _cim22_protoc_configs)
      string(TOUPPER "${_cim22_config}" _cim22_config)
      get_target_property(_cim22_protoc protobuf::protoc "IMPORTED_LOCATION_${_cim22_config}")
      if(_cim22_protoc)
        break()
      endif()
    endforeach()
  endif()
  if(NOT _cim22_protoc)
    message(FATAL_ERROR "System Protobuf must export a runnable protoc location")
  endif()
  execute_process(COMMAND "${_cim22_protoc}" --version
    RESULT_VARIABLE _cim22_protoc_result OUTPUT_VARIABLE _cim22_protoc_version
    ERROR_VARIABLE _cim22_protoc_error OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT _cim22_protoc_result EQUAL 0 OR
      NOT _cim22_protoc_version STREQUAL "libprotoc 31.1")
    message(FATAL_ERROR
      "System protoc must be 31.1: ${_cim22_protoc_version} ${_cim22_protoc_error}")
  endif()
  set(MLIRCIM22_ONNX_PROTO_TARGET ONNX::onnx_proto)
else()
  # Let ONNX own its compatible protoc, headers, runtime, and transitive deps.
  set(ONNX_BUILD_CUSTOM_PROTOBUF ON CACHE BOOL "" FORCE)
  set(ONNX_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
  set(ONNX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(ONNX_GEN_PB_TYPE_STUBS OFF CACHE BOOL "" FORCE)
  set(ONNX_INSTALL OFF CACHE BOOL "" FORCE)
  set(protobuf_FORCE_FETCH_DEPENDENCIES ON CACHE BOOL "" FORCE)

  FetchContent_Declare(onnx
    URL https://github.com/onnx/onnx/archive/refs/tags/v1.21.0.tar.gz
    URL_HASH
      SHA256=42ffedcd8c9b6363694300c6ffec1ada77f9620176465719acb27b13a4d6f2de
    DOWNLOAD_EXTRACT_TIMESTAMP FALSE
    SYSTEM
    EXCLUDE_FROM_ALL
  )
  FetchContent_MakeAvailable(onnx)
  FetchContent_GetProperties(protobuf SOURCE_DIR _cim22_protobuf_source)
  include("${_cim22_protobuf_source}/cmake/protobuf-generate.cmake")
  set(MLIRCIM22_ONNX_PROTO_TARGET onnx_proto)
endif()

if(NOT TARGET protobuf::protoc OR NOT TARGET protobuf::libprotobuf-lite OR
    NOT COMMAND protobuf_generate)
  message(FATAL_ERROR "ONNX must provide Protobuf 31.1 protoc/lite/codegen targets")
endif()
get_target_property(_cim22_lite_type protobuf::libprotobuf-lite TYPE)
if(NOT _cim22_lite_type STREQUAL "STATIC_LIBRARY")
  message(FATAL_ERROR "CIM22 deployment requires static protobuf::libprotobuf-lite")
endif()

if(NOT TARGET ${MLIRCIM22_ONNX_PROTO_TARGET})
  message(FATAL_ERROR
    "ONNX dependency does not provide ${MLIRCIM22_ONNX_PROTO_TARGET}"
  )
endif()

message(STATUS
  "mlir-cim22 ONNX dependency: ${MLIRCIM22_ONNX_PROTO_TARGET}"
)
