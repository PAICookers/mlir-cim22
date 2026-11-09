//===- UsbDriver.h - Native USB driver boundary -------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIM22_RUNTIME_USBDRIVER_H
#define CIM22_RUNTIME_USBDRIVER_H

#include "CIM22/Runtime/CIMDevice.h"
#include "CIM22/Runtime/UsbCIMDevice.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace cim22::runtime::detail {

// Platform drivers expose only byte-stream lifecycle and transfers. FPGA
// records, CIM flits, completion policy, and transaction faults remain in the
// portable UsbCIMDevice implementation.
class UsbDriver {
public:
  virtual ~UsbDriver() = default;

  // Success means the initial IN request is already submitted, before any OUT.
  // Driver-owned buffers remain valid through their completion callback.
  virtual llvm::Error open(const UsbCIMDeviceConfig &config) = 0;
  virtual llvm::Expected<size_t> write(llvm::ArrayRef<uint8_t> bytes) = 0;
  virtual llvm::Expected<size_t> read(llvm::MutableArrayRef<uint8_t> bytes) = 0;
  // Request shutdown without releasing outstanding transfer buffers; drain
  // collects tail completions after the reader joins. Errors remain observable
  // until close, which waits before releasing driver-owned buffers.
  virtual void cancel() = 0;
  // After cancel and joining the reader, return completed tail bytes until
  // zero, or a real receive error. An expected cancellation is not an error.
  // Synchronous drivers have no outstanding request once the reader joins.
  virtual llvm::Expected<size_t> drain(llvm::MutableArrayRef<uint8_t>) {
    return size_t{0};
  }
  // Call drain before close to observe shutdown completions and errors.
  virtual void close() = 0;
};

std::unique_ptr<UsbDriver> createPlatformUsbDriver();

// Test-only seam for protocol and lifecycle verification without hardware.
std::unique_ptr<CIMDevice>
createUsbCIMDevice(const UsbCIMDeviceConfig &config,
                   std::unique_ptr<UsbDriver> driver);

} // namespace cim22::runtime::detail

#endif // CIM22_RUNTIME_USBDRIVER_H
