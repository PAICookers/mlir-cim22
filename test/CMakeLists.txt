find_package(Python3 REQUIRED COMPONENTS Interpreter)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

execute_process(
  COMMAND ${Python3_EXECUTABLE} -c
    "import numpy, onnx; from onnx import checker, helper, numpy_helper, shape_inference; from onnx.reference import ReferenceEvaluator; assert onnx.__version__ == '1.21.0'"
  RESULT_VARIABLE MLIRCIM22_PYTHON_TEST_DEPS_RESULT
  ERROR_VARIABLE MLIRCIM22_PYTHON_TEST_DEPS_ERROR
)
if(NOT MLIRCIM22_PYTHON_TEST_DEPS_RESULT EQUAL 0)
  message(FATAL_ERROR
    "mlir-cim22 tests require the locked uv test environment for "
    "${Python3_EXECUTABLE}: ${MLIRCIM22_PYTHON_TEST_DEPS_ERROR}"
  )
endif()

add_executable(mlir-cim22-int8-weight-layout-test EXCLUDE_FROM_ALL
  Unit/Support/Int8WeightLayoutTest.cpp
)
target_link_libraries(mlir-cim22-int8-weight-layout-test PRIVATE
  MLIRCIM22Support
)
set_target_properties(mlir-cim22-int8-weight-layout-test PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${LLVM_RUNTIME_OUTPUT_INTDIR}
)

add_executable(mlir-cim22-matmulinteger-importer-test EXCLUDE_FROM_ALL
  Unit/Frontend/ONNX/MatMulIntegerImporterTest.cpp
)
target_link_libraries(mlir-cim22-matmulinteger-importer-test PRIVATE
  MLIRCIM22LinalgToCIM
  MLIRCIM22ONNXFrontend
)
set_target_properties(mlir-cim22-matmulinteger-importer-test PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${LLVM_RUNTIME_OUTPUT_INTDIR}
)

add_executable(mlir-cim22-cimframe-codec-test EXCLUDE_FROM_ALL
  Unit/Target/CIMFrame/CIMFrameCodecTest.cpp
)
target_link_libraries(mlir-cim22-cimframe-codec-test PRIVATE
  MLIRCIM22Target
)
set_target_properties(mlir-cim22-cimframe-codec-test PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${LLVM_RUNTIME_OUTPUT_INTDIR}
)

set(MLIRCIM22_TEST_DEPENDS
  FileCheck
  mlir-cim22-cimframe-codec-test
  mlir-cim22-matmulinteger-importer-test
  mlir-cim22-int8-weight-layout-test
  mlir-cim22-onnx-import
  mlir-cim22-opt
  split-file
)

add_lit_testsuite(check-mlir-cim22 "Running mlir-cim22 regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${MLIRCIM22_TEST_DEPENDS}
)
set_target_properties(check-mlir-cim22 PROPERTIES FOLDER "Tests")

add_lit_testsuites(MLIRCIM22 ${CMAKE_CURRENT_SOURCE_DIR}
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${MLIRCIM22_TEST_DEPENDS}
)
