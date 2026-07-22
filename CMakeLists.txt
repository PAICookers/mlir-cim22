cmake_minimum_required(VERSION 3.20.0)

if(POLICY CMP0219)
  cmake_policy(SET CMP0219 NEW)
endif()

project(MLIRCIM22 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(MLIR REQUIRED CONFIG)

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
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(lib/Support)
add_subdirectory(include/CIM22/Dialect/CIM/IR)
add_subdirectory(include/CIM22/Dialect/CIMFrame/IR)
add_subdirectory(include/CIM22/Dialect/CIMFrame/Transforms)
add_subdirectory(lib/Dialect/CIM/IR)
add_subdirectory(lib/Dialect/CIMFrame/IR)
add_subdirectory(lib/Dialect/CIMFrame/Transforms)
add_subdirectory(tools/mlir-cim22-opt)

option(MLIRCIM22_INCLUDE_TESTS "Build mlir-cim22 tests" ON)
if(MLIRCIM22_INCLUDE_TESTS)
  add_subdirectory(test)
endif()
