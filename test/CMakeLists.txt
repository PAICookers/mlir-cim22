configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(MLIRCIM22_TEST_DEPENDS
  FileCheck
  mlir-cim22-opt
)

add_lit_testsuite(check-mlir-cim22 "Running mlir-cim22 regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${MLIRCIM22_TEST_DEPENDS}
)
set_target_properties(check-mlir-cim22 PROPERTIES FOLDER "Tests")

add_lit_testsuites(MLIRCIM22 ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${MLIRCIM22_TEST_DEPENDS}
)
