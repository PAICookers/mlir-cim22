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
  DEPENDS ${MLIRCIM22_TEST_DEPENDS}
)
