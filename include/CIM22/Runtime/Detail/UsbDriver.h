//===- UsbDriver.h - Native USB driver boundary -------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIM22_RUNTIME_DETAIL_USBDRIVER_H
#define CIM22_RUNTIME_DETAIL_USBDRIVER_H

#include "CIM22/Runtime/CIMDevice.h"
#include "CIM22/Runtime/UsbCIMDevice.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace cim22::runtime::detail {

// Platform drivers expose only byte-stream lifecycle and transfers. FPGA
// records, CIM flits, completion policy, and sticky faults remain in the
// portable UsbCIMDevice implementation.
class UsbDriver {
public:
  virtual ~UsbDriver() = default;

  virtual llvm::Error open(const UsbCIMDeviceConfig &config) = 0;
  virtual llvm::Expected<size_t> write(llvm::ArrayRef<uint8_t> bytes) = 0;
  virtual llvm::Expected<size_t> read(llvm::MutableArrayRef<uint8_t> bytes) = 0;
  virtual void cancel() = 0;
  virtual void close() = 0;
};

std::unique_ptr<UsbDriver> createPlatformUsbDriver();

// Test-only seam for protocol and lifecycle verification without hardware.
std::unique_ptr<CIMDevice>
createUsbCIMDevice(const UsbCIMDeviceConfig &config,
                   std::unique_ptr<UsbDriver> driver);

} // namespace cim22::runtime::detail

#endif // CIM22_RUNTIME_DETAIL_USBDRIVER_H
