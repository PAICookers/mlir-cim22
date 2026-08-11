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
  set(MLIRCIM22_ONNX_PROTO_TARGET onnx_proto)
endif()

if(NOT TARGET ${MLIRCIM22_ONNX_PROTO_TARGET})
  message(FATAL_ERROR
    "ONNX dependency does not provide ${MLIRCIM22_ONNX_PROTO_TARGET}"
  )
endif()

message(STATUS
  "mlir-cim22 ONNX dependency: ${MLIRCIM22_ONNX_PROTO_TARGET}"
)
