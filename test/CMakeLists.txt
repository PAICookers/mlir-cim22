configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

add_executable(mlir-cim22-int8-weight-layout-test EXCLUDE_FROM_ALL
  Support/Int8WeightLayoutTest.cpp
)
target_link_libraries(mlir-cim22-int8-weight-layout-test PRIVATE
  MLIRCIM22Support
)
set_target_properties(mlir-cim22-int8-weight-layout-test PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${LLVM_RUNTIME_OUTPUT_INTDIR}
)

set(MLIRCIM22_TEST_DEPENDS
  FileCheck
  mlir-cim22-int8-weight-layout-test
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
  SKIP "^Conversion$" "^Dialect/CIM$"
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
get_subproject_title(mlircim22_subproject_title)
set_target_properties(
  check-mlircim22-conversion
  check-mlircim22-dialect-cim
  PROPERTIES
  FOLDER "${mlircim22_subproject_title}/Tests/LIT Testsuites"
)
