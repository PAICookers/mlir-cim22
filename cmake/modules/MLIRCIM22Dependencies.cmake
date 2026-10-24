#===- MLIRCIM22Dependencies.cmake - Third-party dependencies --------------===#
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===#

include_guard(GLOBAL)
include(FetchContent)

set(protobuf_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
# Match LLVM and the DLL ABI: allocations may cross the runtime boundary.
set(protobuf_MSVC_STATIC_RUNTIME OFF CACHE BOOL "" FORCE)
set(protobuf_BUILD_TESTS OFF CACHE BOOL "" FORCE)
# protoc requires libupb even though deployed models only use lite.
set(protobuf_BUILD_LIBUPB ON CACHE BOOL "" FORCE)
set(protobuf_WITH_ZLIB OFF CACHE BOOL "" FORCE)
set(protobuf_INSTALL OFF CACHE BOOL "" FORCE)
# Keep independently cached dependency exports disabled when reconfiguring.
set(utf8_range_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(ABSL_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(protobuf_FORCE_FETCH_DEPENDENCIES ON CACHE BOOL "" FORCE)
FetchContent_Declare(protobuf
  URL https://github.com/protocolbuffers/protobuf/releases/download/v31.1/protobuf-31.1.tar.gz
  URL_HASH
    SHA256=12bfd76d27b9ac3d65c00966901609e020481b9474ef75c7ff4601ac06fa0b82
  DOWNLOAD_EXTRACT_TIMESTAMP FALSE
  SYSTEM
  EXCLUDE_FROM_ALL)
FetchContent_MakeAvailable(protobuf)
include("${protobuf_SOURCE_DIR}/cmake/protobuf-generate.cmake")

if(MLIRCIM22_ENABLE_ONNX)
  # ONNX reuses the existing Protobuf targets and their matching protoc.
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
    EXCLUDE_FROM_ALL)
  FetchContent_MakeAvailable(onnx)
  set(MLIRCIM22_ONNX_PROTO_TARGET onnx_proto)
endif()
