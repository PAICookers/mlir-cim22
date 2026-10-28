set(_cim22_proto "${PROJECT_SOURCE_DIR}/schema/cim22_model.proto")
set(_cim22_generated "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${_cim22_generated}")
protobuf_generate(LANGUAGE cpp OUT_VAR _cim22_proto_cpp
  PROTOS "${_cim22_proto}" IMPORT_DIRS "${PROJECT_SOURCE_DIR}/schema"
  PROTOC_OUT_DIR "${_cim22_generated}" DEPENDENCIES protobuf::protoc)
protobuf_generate(LANGUAGE python OUT_VAR _cim22_proto_python
  PROTOS "${_cim22_proto}" IMPORT_DIRS "${PROJECT_SOURCE_DIR}/schema"
  PROTOC_OUT_DIR "${_cim22_generated}" DEPENDENCIES protobuf::protoc)
add_custom_target(check-cim22-protobuf
  COMMAND ${CMAKE_COMMAND} -E compare_files "${_cim22_proto_python}"
    "${PROJECT_SOURCE_DIR}/python/tests/_generated/cim22_model_pb2.py"
  DEPENDS ${_cim22_proto_python}
  COMMENT "Checking test Python Protobuf binding freshness" VERBATIM)
add_custom_target(update-cim22-protobuf
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_cim22_proto_python}"
    "${PROJECT_SOURCE_DIR}/python/tests/_generated/cim22_model_pb2.py"
  DEPENDS ${_cim22_proto_python}
  COMMENT "Regenerating test Python Protobuf binding" VERBATIM)

# Execution-only deployment library; Protobuf lite and LLVMSupport are static.
option(CIM22_ENABLE_USB "Build the platform USB runtime" ON)
if(CIM22_ENABLE_USB)
  if(WIN32)
    set(CYAPI_ROOT "" CACHE PATH "Root of the Cypress EZ-USB FX3 SDK")
    find_path(CYAPI_INCLUDE_DIR CyAPI.h HINTS "${CYAPI_ROOT}"
      PATH_SUFFIXES library/cpp/inc include REQUIRED)
    find_library(CYAPI_LIBRARY NAMES CyAPI HINTS "${CYAPI_ROOT}"
      PATH_SUFFIXES library/cpp/lib/x64 lib/x64 x64 REQUIRED)
    # Catch a supplied x86 library or incompatible SDK/CRT before a long build.
    include(CheckCXXSourceCompiles)
    include(CMakePushCheckState)
    block()
      cmake_push_check_state(RESET)
      set(CMAKE_REQUIRED_INCLUDES "${CYAPI_INCLUDE_DIR}")
      set(CMAKE_REQUIRED_LIBRARIES "${CYAPI_LIBRARY}" setupapi legacy_stdio_definitions)
      # Match the selected CRT instead of probing a Release SDK in Debug mode.
      set(CMAKE_TRY_COMPILE_CONFIGURATION Release)
      if(CMAKE_BUILD_TYPE)
        set(CMAKE_TRY_COMPILE_CONFIGURATION "${CMAKE_BUILD_TYPE}")
      endif()
      unset(CIM22_CYAPI_LINKS CACHE)
      check_cxx_source_compiles("
        #include <windows.h>
        #include <CyAPI.h>
        int main() { CCyUSBDevice device(nullptr, CYUSBDRV_GUID, false);
                     return device.DeviceCount(); }" CIM22_CYAPI_LINKS)
      cmake_pop_check_state()
    endblock()
    if(NOT CIM22_CYAPI_LINKS)
      message(FATAL_ERROR "CyAPI could not link with the Windows toolchain. "
        "Provide matching x64 headers and CyAPI.lib via CYAPI_ROOT or "
        "CYAPI_INCLUDE_DIR/CYAPI_LIBRARY; x86 CyAPI.lib cannot build win_amd64.")
    endif()
    add_library(cim22_usb_driver STATIC IMPORTED)
    set_target_properties(cim22_usb_driver PROPERTIES
      IMPORTED_LOCATION "${CYAPI_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${CYAPI_INCLUDE_DIR}"
      INTERFACE_LINK_LIBRARIES "setupapi;legacy_stdio_definitions")
    set(_cim22_usb_driver_source CyApiUsbDriver.cpp)
  else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(LIBUSB QUIET IMPORTED_TARGET GLOBAL libusb-1.0)
    if(NOT LIBUSB_FOUND)
      message(FATAL_ERROR
        "Linux USB support requires libusb-1.0 development files, not the Cypress "
        "SDK. Install libusb-1.0-0-dev (Debian/Ubuntu), libusb (Arch), or "
        "libusb1-devel (Fedora), or set CIM22_ENABLE_USB=OFF for software only.")
    endif()
    add_library(cim22_usb_driver ALIAS PkgConfig::LIBUSB)
    set(_cim22_usb_driver_source LibusbUsbDriver.cpp)
  endif()
endif()

set(_cim22_runtime_sources Runtime.cpp ModelData.cpp CIMBoardRunner.cpp)
if(CIM22_ENABLE_USB)
  find_package(Threads REQUIRED)
  list(APPEND _cim22_runtime_sources UsbCIMDevice.cpp UsbRecordCodec.cpp
       ${_cim22_usb_driver_source} PLLSerial.cpp)
endif()
add_library(cim22_runtime SHARED ${_cim22_runtime_sources}
  ${_cim22_proto_cpp}
  $<TARGET_OBJECTS:obj.MLIRCIM22Execution>
  $<TARGET_OBJECTS:obj.MLIRCIM22Support>)
llvm_update_compile_flags(cim22_runtime)
target_include_directories(cim22_runtime PRIVATE "${_cim22_generated}")
target_link_libraries(cim22_runtime PRIVATE LLVMSupport protobuf::libprotobuf-lite
  ${CMAKE_DL_LIBS})
if(CIM22_ENABLE_USB)
  target_compile_definitions(cim22_runtime PRIVATE CIM22_HAS_USB=1)
  target_link_libraries(cim22_runtime PRIVATE cim22_usb_driver Threads::Threads)
  if(WIN32)
    # CyAPI includes Windows headers; keep their min/max macros out of C++.
    target_compile_definitions(cim22_runtime PRIVATE NOMINMAX)
  endif()
endif()

# ExecutionEngine is excluded from libMLIR; its exported target carries the
# native codegen/JIT dependencies for either static or shared LLVM/MLIR builds.
add_library(cim22_compiler SHARED Compiler.cpp)
llvm_update_compile_flags(cim22_compiler)
target_link_libraries(cim22_compiler PRIVATE
  cim22_runtime MLIRCIM22Compiler MLIRExecutionEngine)
mlir_target_link_libraries(cim22_compiler PRIVATE
  MLIRBuiltinToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
  MLIRTargetLLVMIRExport)
set_target_properties(cim22_runtime cim22_compiler PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${LLVM_LIBRARY_OUTPUT_INTDIR}"
  LIBRARY_OUTPUT_DIRECTORY "${LLVM_LIBRARY_OUTPUT_INTDIR}"
  RUNTIME_OUTPUT_DIRECTORY "${LLVM_LIBRARY_OUTPUT_INTDIR}"
  WINDOWS_EXPORT_ALL_SYMBOLS ON)

# Keep the compiler adapter's runtime dependency local to the wheel.
if(SKBUILD)
  set_target_properties(cim22_runtime cim22_compiler PROPERTIES
    INSTALL_RPATH "$ORIGIN")
  install(TARGETS cim22_runtime cim22_compiler
    LIBRARY DESTINATION lib COMPONENT runtime
    RUNTIME DESTINATION lib COMPONENT runtime
    ARCHIVE DESTINATION lib COMPONENT runtime)
  if(WIN32 AND CIM22_ENABLE_USB)
    find_file(CYAPI_LICENSE_FILE license.rtf
      HINTS "${CYAPI_ROOT}" "${CYAPI_INCLUDE_DIR}/../../.."
      PATH_SUFFIXES license REQUIRED)
    install(FILES "${CYAPI_LICENSE_FILE}"
      DESTINATION share/licenses/cyapi COMPONENT runtime)
  endif()
endif()
