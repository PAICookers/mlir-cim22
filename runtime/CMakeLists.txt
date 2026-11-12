# Execution-only deployment library; Protobuf lite and LLVMSupport are static.
option(CIM22_ENABLE_USB "Build the platform USB runtime" OFF)
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
    pkg_check_modules(LIBUSB REQUIRED IMPORTED_TARGET GLOBAL libusb-1.0)
    set(CYUSB_ROOT "${PROJECT_SOURCE_DIR}/third_party/cyusb-linux" CACHE PATH
      "Root of the Cypress CyUSB Linux SDK (contains include/ and lib/)")
    set(CYUSB_INCLUDE_DIR "" CACHE PATH
      "Directory containing cyusb.h")
    set(CYUSB_LIBRARY "" CACHE FILEPATH
      "Path to the CyUSB Linux libcyusb shared library")
    if(NOT CYUSB_INCLUDE_DIR)
      unset(CYUSB_INCLUDE_DIR CACHE)
    endif()
    if(NOT CYUSB_LIBRARY)
      unset(CYUSB_LIBRARY CACHE)
    endif()
    find_path(CYUSB_INCLUDE_DIR cyusb.h
      HINTS "${CYUSB_ROOT}"
      PATH_SUFFIXES include)
    find_library(CYUSB_LIBRARY NAMES cyusb libcyusb
      HINTS "${CYUSB_ROOT}"
      PATH_SUFFIXES lib)
    if(NOT CYUSB_INCLUDE_DIR OR NOT CYUSB_LIBRARY)
      message(FATAL_ERROR
        "Linux USB support requires the Cypress CyUSB Linux SDK (libcyusb). "
        "Set CYUSB_ROOT, or set CYUSB_INCLUDE_DIR and CYUSB_LIBRARY explicitly; "
        "do not run the SDK install.sh. Set CIM22_ENABLE_USB=OFF for software only.")
    endif()
    set(CYUSB_LICENSE_FILE "" CACHE FILEPATH
      "Path to the CyUSB Linux license text for wheel installation")
    if(NOT CYUSB_LICENSE_FILE
       AND CYUSB_ROOT STREQUAL "${PROJECT_SOURCE_DIR}/third_party/cyusb-linux"
       AND EXISTS "${CYUSB_ROOT}/license/license.txt")
      set(CYUSB_LICENSE_FILE "${CYUSB_ROOT}/license/license.txt"
        CACHE FILEPATH "Path to the CyUSB Linux license text for wheel installation"
        FORCE)
    endif()
    add_library(cim22_usb_driver SHARED IMPORTED GLOBAL)
    set_target_properties(cim22_usb_driver PROPERTIES
      IMPORTED_LOCATION "${CYUSB_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${CYUSB_INCLUDE_DIR}"
      INTERFACE_LINK_LIBRARIES "PkgConfig::LIBUSB")
    set(_cim22_usb_driver_source LibcyusbUsbDriver.cpp)
  endif()
endif()

set(_cim22_runtime_sources Runtime.cpp CIMBoardRunner.cpp)
if(CIM22_ENABLE_USB)
  find_package(Threads REQUIRED)
  list(APPEND _cim22_runtime_sources UsbCIMDevice.cpp UsbRecordCodec.cpp
       ${_cim22_usb_driver_source})
endif()
add_library(cim22_runtime SHARED ${_cim22_runtime_sources}
  $<TARGET_OBJECTS:obj.MLIRCIM22Execution>
  $<TARGET_OBJECTS:obj.MLIRCIM22Support>)
llvm_update_compile_flags(cim22_runtime)
target_include_directories(cim22_runtime PRIVATE
  "${PROJECT_SOURCE_DIR}/runtime/include"
  "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(cim22_runtime PRIVATE MLIRCIM22RuntimeModel LLVMSupport
  ${CMAKE_DL_LIBS})
if(WIN32)
  # Keep Windows min/max macros out of the runtime sources that use std::min.
  target_compile_definitions(cim22_runtime PRIVATE NOMINMAX)
endif()
if(CIM22_ENABLE_USB)
  target_compile_definitions(cim22_runtime PRIVATE CIM22_HAS_USB=1)
  target_link_libraries(cim22_runtime PRIVATE cim22_usb_driver Threads::Threads)
endif()

# Keep the device runtime independent from the compiler tool.
set_target_properties(cim22_runtime PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${LLVM_LIBRARY_OUTPUT_INTDIR}"
  LIBRARY_OUTPUT_DIRECTORY "${LLVM_LIBRARY_OUTPUT_INTDIR}"
  RUNTIME_OUTPUT_DIRECTORY "${LLVM_LIBRARY_OUTPUT_INTDIR}"
  WINDOWS_EXPORT_ALL_SYMBOLS ON)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  # LLVM/MLIR internals must not interpose TensorFlow or other compiler wheels.
  foreach(_cim22_library IN ITEMS cim22_runtime)
    target_link_options(${_cim22_library} PRIVATE
      "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/cim22.exports")
    set_property(TARGET ${_cim22_library} APPEND PROPERTY LINK_DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/cim22.exports")
  endforeach()
  # Do not embed the developer's absolute SDK path in build artifacts.
  set_target_properties(cim22_runtime PROPERTIES
    SKIP_BUILD_RPATH TRUE)
  unset(_cim22_library)
endif()

# Keep the runtime model dependency local to the wheel.
if(SKBUILD)
  set_target_properties(cim22_runtime PROPERTIES
    INSTALL_RPATH "$ORIGIN")
  install(TARGETS cim22_runtime
    LIBRARY DESTINATION lib COMPONENT runtime
    RUNTIME DESTINATION lib COMPONENT runtime
    ARCHIVE DESTINATION lib COMPONENT runtime)
  if(WIN32 AND CIM22_ENABLE_USB)
    find_file(CYAPI_LICENSE_FILE license.rtf
      HINTS "${CYAPI_ROOT}" "${CYAPI_INCLUDE_DIR}/../../.."
      PATH_SUFFIXES license REQUIRED)
    install(FILES "${CYAPI_LICENSE_FILE}"
      DESTINATION share/licenses/cyapi COMPONENT runtime)
  elseif(UNIX AND CIM22_ENABLE_USB)
    install(FILES "${CYUSB_LIBRARY}" DESTINATION lib COMPONENT runtime)
    if(CYUSB_LICENSE_FILE AND EXISTS "${CYUSB_LICENSE_FILE}")
      install(FILES "${CYUSB_LICENSE_FILE}"
        DESTINATION share/licenses/cyusb-linux COMPONENT runtime)
    endif()
  endif()
endif()
