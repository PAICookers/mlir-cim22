#===- MLIRCIM22Dependencies.cmake - Third-party dependencies --------------===#
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===#

include_guard(GLOBAL)

include(FetchContent)

find_package(Protobuf CONFIG REQUIRED)
if(Protobuf_VERSION VERSION_LESS 4.25.1)
  message(FATAL_ERROR
    "ONNX v1.21.0 requires Protobuf 4.25.1 or newer; found "
    "${Protobuf_VERSION}"
  )
endif()

option(MLIRCIM22_USE_SYSTEM_ONNX
  "Use an installed ONNX package instead of the pinned source dependency"
  OFF
)

if(MLIRCIM22_USE_SYSTEM_ONNX)
  find_package(ONNX 1.21.0 EXACT CONFIG REQUIRED)
  set(MLIRCIM22_ONNX_PROTO_TARGET ONNX::onnx_proto)
else()
  # Keep the production parser tied to one reviewed ONNX schema revision.
  set(ONNX_BUILD_CUSTOM_PROTOBUF OFF CACHE BOOL "" FORCE)
  set(ONNX_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
  set(ONNX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(ONNX_GEN_PB_TYPE_STUBS OFF CACHE BOOL "" FORCE)
  set(ONNX_INSTALL OFF CACHE BOOL "" FORCE)

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

add_library(MLIRCIM22ONNXProto INTERFACE)
add_library(MLIRCIM22::ONNXProto ALIAS MLIRCIM22ONNXProto)
target_link_libraries(MLIRCIM22ONNXProto
  INTERFACE ${MLIRCIM22_ONNX_PROTO_TARGET}
)
