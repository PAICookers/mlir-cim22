cmake_minimum_required(VERSION 3.28.0)

if(POLICY CMP0219)
  cmake_policy(SET CMP0219 NEW)
endif()

project(MLIRCIM22 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(MLIR REQUIRED CONFIG)

option(MLIRCIM22_INCLUDE_TESTS "Build mlir-cim22 tests" OFF)
option(MLIRCIM22_ENABLE_ONNX "Build the ONNX importer" ${MLIRCIM22_INCLUDE_TESTS})
if(MLIRCIM22_INCLUDE_TESTS AND NOT MLIRCIM22_ENABLE_ONNX)
  message(FATAL_ERROR "MLIRCIM22_INCLUDE_TESTS requires MLIRCIM22_ENABLE_ONNX")
endif()

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)
include(MLIRCIM22Dependencies)

set(LLVM_LIT_ARGS "-sv" CACHE STRING "Default options for lit")
set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH ${MLIR_CMAKE_DIR} ${LLVM_CMAKE_DIR})
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

set(MLIRCIM22_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(MLIRCIM22_BINARY_DIR ${PROJECT_BINARY_DIR})

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)

if(MLIRCIM22_INCLUDE_TESTS)
  add_subdirectory(test)
endif()
