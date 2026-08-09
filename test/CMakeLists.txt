configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

find_package(Python3 REQUIRED COMPONENTS Interpreter)
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c
    "import numpy, onnx; assert onnx.__version__ == '1.21.0'"
  RESULT_VARIABLE MLIRCIM22_PYTHON_TEST_DEPS_RESULT
  ERROR_VARIABLE MLIRCIM22_PYTHON_TEST_DEPS_ERROR
)
if(NOT MLIRCIM22_PYTHON_TEST_DEPS_RESULT EQUAL 0)
  message(FATAL_ERROR
    "mlir-cim22 tests require the packages in requirements-test.txt for "
    "${Python3_EXECUTABLE}: ${MLIRCIM22_PYTHON_TEST_DEPS_ERROR}"
  )
endif()

add_executable(mlir-cim22-int8-weight-layout-test EXCLUDE_FROM_ALL
  Support/Int8WeightLayoutTest.cpp
)
target_link_libraries(mlir-cim22-int8-weight-layout-test PRIVATE
  MLIRCIM22Support
)
set_target_properties(mlir-cim22-int8-weight-layout-test PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${LLVM_RUNTIME_OUTPUT_INTDIR}
)

add_executable(mlir-cim22-f0-importer-test EXCLUDE_FROM_ALL
  Frontend/ONNX/F0ImporterTest.cpp
)
target_link_libraries(mlir-cim22-f0-importer-test PRIVATE
  MLIRCIM22LinalgToCIM
  MLIRCIM22ONNXFrontend
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRPass
  MLIRTensorDialect
)
set_target_properties(mlir-cim22-f0-importer-test PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${LLVM_RUNTIME_OUTPUT_INTDIR}
)

add_executable(mlir-cim22-program-image-test EXCLUDE_FROM_ALL
  Artifact/ProgramImageTest.cpp
)
llvm_update_compile_flags(mlir-cim22-program-image-test)
target_link_libraries(mlir-cim22-program-image-test PRIVATE
  MLIRCIM22Artifact
)
set_target_properties(mlir-cim22-program-image-test PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${LLVM_RUNTIME_OUTPUT_INTDIR}
)

set(MLIRCIM22_TEST_DEPENDS
  FileCheck
  mlir-cim22-f0-importer-test
  mlir-cim22-int8-weight-layout-test
  mlir-cim22-onnx-import
  mlir-cim22-opt
  mlir-cim22-program-image-test
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
  SKIP "^Conversion($|/)" "^Dialect/CIM$" "^Target($|/)" "__pycache__"
)

add_lit_target(check-mlircim22-conversion
  "Running lit suite ${CMAKE_CURRENT_SOURCE_DIR}/Conversion"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${MLIRCIM22_TEST_DEPENDS}
  ARGS --filter=Conversion/
)
add_lit_target(check-mlircim22-dialect-cim
  "Running lit suite ${CMAKE_CURRENT_SOURCE_DIR}/Dialect/CIM"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${MLIRCIM22_TEST_DEPENDS}
  ARGS --filter=Dialect/CIM/
)
add_lit_target(check-mlircim22-target
  "Running lit suite ${CMAKE_CURRENT_SOURCE_DIR}/Target"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS FileCheck mlir-cim22-opt
  ARGS --filter=Target/
)
get_subproject_title(mlircim22_subproject_title)
set_target_properties(
  check-mlircim22-conversion
  check-mlircim22-dialect-cim
  check-mlircim22-target
  PROPERTIES
  FOLDER "${mlircim22_subproject_title}/Tests/LIT Testsuites"
)
