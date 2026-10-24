find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(CIM22_HAS_USB 0)
if(CIM22_ENABLE_USB)
  set(CIM22_HAS_USB 1)
endif()

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

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${LLVM_RUNTIME_OUTPUT_INTDIR})
# Test assertions must execute even when the compiler is built in Release.
add_compile_options(-UNDEBUG)

add_executable(mlir-cim22-model-data-test EXCLUDE_FROM_ALL
  Unit/Execution/ModelDataTest.cpp)
llvm_update_compile_flags(mlir-cim22-model-data-test)
target_include_directories(mlir-cim22-model-data-test PRIVATE
  "${PROJECT_SOURCE_DIR}/lib/Runtime")
target_link_libraries(mlir-cim22-model-data-test PRIVATE cim22_runtime LLVMSupport)

add_executable(mlir-cim22-int8-weight-layout-test EXCLUDE_FROM_ALL
  Unit/Support/Int8WeightLayoutTest.cpp
)
target_link_libraries(mlir-cim22-int8-weight-layout-test PRIVATE
  MLIRCIM22Support
)
add_executable(mlir-cim22-bf16-support-test EXCLUDE_FROM_ALL
  Unit/Support/BF16SupportTest.cpp
)
target_link_libraries(mlir-cim22-bf16-support-test PRIVATE
  MLIRCIM22Support
)
add_executable(mlir-cim22-matmulinteger-importer-test EXCLUDE_FROM_ALL
  Unit/Frontend/ONNX/MatMulIntegerImporterTest.cpp
)
target_link_libraries(mlir-cim22-matmulinteger-importer-test PRIVATE
  MLIRCIM22LinalgToCIM
  MLIRCIM22ONNXFrontend
)
add_executable(mlir-cim22-cimframe-codec-test EXCLUDE_FROM_ALL
  Unit/Target/CIMFrame/CIMFrameCodecTest.cpp
)
target_link_libraries(mlir-cim22-cimframe-codec-test PRIVATE
  MLIRCIM22Target
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRParser
  MLIRTensorDialect
)
add_executable(mlir-cim22-cim-executable-test EXCLUDE_FROM_ALL
  Unit/Target/CIMExecutableTest.cpp
)
target_link_libraries(mlir-cim22-cim-executable-test PRIVATE
  MLIRCIM22Target
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRParser
  MLIRTensorDialect
)
add_executable(mlir-cim22-cim-software-runner-test EXCLUDE_FROM_ALL
  Unit/Execution/CIMSoftwareRunnerTest.cpp
)
target_link_libraries(mlir-cim22-cim-software-runner-test PRIVATE
  MLIRCIM22Execution
  MLIRCIM22Target
)
target_compile_options(mlir-cim22-cim-software-runner-test PRIVATE -fno-rtti)
add_executable(mlir-cim22-cim-executable-runner-test EXCLUDE_FROM_ALL
  Unit/Execution/CIMExecutableRunnerIntegrationTest.cpp
)
target_link_libraries(mlir-cim22-cim-executable-runner-test PRIVATE
  MLIRCIM22Execution
  MLIRCIM22Target
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRParser
  MLIRTensorDialect
)
target_compile_options(mlir-cim22-cim-executable-runner-test PRIVATE -fno-rtti)
add_executable(mlir-cim22-cim-device-runtime-test EXCLUDE_FROM_ALL
  Unit/Execution/CIMDeviceRuntimeTest.cpp)
llvm_update_compile_flags(mlir-cim22-cim-device-runtime-test)
target_link_libraries(mlir-cim22-cim-device-runtime-test PRIVATE
  cim22_runtime MLIRCIM22Target)
target_compile_options(mlir-cim22-cim-device-runtime-test PRIVATE -fno-rtti)
if(CIM22_ENABLE_USB)
  add_executable(mlir-cim22-usb-runtime-test EXCLUDE_FROM_ALL
    Unit/Execution/UsbCIMDeviceTest.cpp)
  llvm_update_compile_flags(mlir-cim22-usb-runtime-test)
  target_include_directories(mlir-cim22-usb-runtime-test PRIVATE
    "${PROJECT_SOURCE_DIR}/lib/Runtime")
  target_link_libraries(mlir-cim22-usb-runtime-test PRIVATE cim22_runtime)
  if(WIN32)
    # The fake driver exchanges llvm::Error with USB internals. Static LLVM
    # type IDs are distinct across Windows DLLs, so keep this C++ test seam
    # in one binary; calls to the runtime's C ABI still use the real DLL.
    target_sources(mlir-cim22-usb-runtime-test PRIVATE
      "${PROJECT_SOURCE_DIR}/lib/Runtime/UsbCIMDevice.cpp"
      "${PROJECT_SOURCE_DIR}/lib/Runtime/UsbRecordCodec.cpp")
    target_link_libraries(mlir-cim22-usb-runtime-test PRIVATE LLVMSupport)
  endif()
  if(MSVC)
    target_compile_options(mlir-cim22-usb-runtime-test PRIVATE /GR-)
  else()
    target_compile_options(mlir-cim22-usb-runtime-test PRIVATE -fno-rtti)
  endif()
endif()
if(WIN32)
  # SDK double exercises the production adapter without hardware or CyAPI.lib.
  add_executable(mlir-cim22-cyapi-driver-test EXCLUDE_FROM_ALL
    Unit/Execution/CyApiUsbDriverTest.cpp
    "${PROJECT_SOURCE_DIR}/lib/Runtime/CyApiUsbDriver.cpp")
  llvm_update_compile_flags(mlir-cim22-cyapi-driver-test)
  target_include_directories(mlir-cim22-cyapi-driver-test BEFORE PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/Unit/Execution/fake-cyapi")
  target_compile_definitions(mlir-cim22-cyapi-driver-test PRIVATE NOMINMAX)
  target_link_libraries(mlir-cim22-cyapi-driver-test PRIVATE LLVMSupport)
endif()
set(MLIRCIM22_TEST_DEPENDS
  cim22_compiler
  check-cim22-protobuf
  mlir-cim22-model-data-test
  FileCheck
  mlir-runner
  mlir_c_runner_utils
  mlir-cim22-bf16-support-test
  mlir-cim22-cimframe-codec-test
  mlir-cim22-cim-executable-test
  mlir-cim22-matmulinteger-importer-test
  mlir-cim22-int8-weight-layout-test
  mlir-cim22-cim-software-runner-test
  mlir-cim22-cim-executable-runner-test
  mlir-cim22-cim-device-runtime-test
  mlir-cim22-onnx-import
  mlir-cim22-opt
  split-file
)
if(TARGET mlir-cim22-usb-runtime-test)
  list(APPEND MLIRCIM22_TEST_DEPENDS mlir-cim22-usb-runtime-test)
endif()
if(TARGET mlir-cim22-cyapi-driver-test)
  list(APPEND MLIRCIM22_TEST_DEPENDS mlir-cim22-cyapi-driver-test)
endif()

add_lit_testsuite(check-mlir-cim22 "Running mlir-cim22 regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${MLIRCIM22_TEST_DEPENDS}
)
set_target_properties(check-mlir-cim22 PROPERTIES FOLDER "Tests")

add_lit_testsuites(MLIRCIM22 ${CMAKE_CURRENT_SOURCE_DIR}
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${MLIRCIM22_TEST_DEPENDS}
)
