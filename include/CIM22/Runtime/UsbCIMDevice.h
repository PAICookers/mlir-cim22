//===- UsbCIMDevice.h - CIM22 USB board backend ------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIM22_RUNTIME_USBCIMDEVICE_H
#define CIM22_RUNTIME_USBCIMDEVICE_H

#include "CIM22/Runtime/CIMDevice.h"
#include "CIM22/Runtime/UsbRecordCodec.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

namespace cim22::runtime {

// Describes the programmed FPGA USB bridge. Endpoint values are supplied by
// the board profile rather than inferred from compiler target semantics.
struct UsbCIMDeviceConfig {
  uint16_t vendorId = 0;
  uint16_t productId = 0;
  // Omit both only when VID/PID identify a unique connected device.
  std::optional<uint8_t> busNumber;
  std::optional<uint8_t> deviceAddress;
  uint8_t interfaceNumber = 0;
  uint8_t alternateSetting = 0;
  uint8_t bulkInEndpoint = 0;
  uint8_t bulkOutEndpoint = 0;
  // These select the Host-to-chip injection path. Response records report
  // their own source link and channel and may differ from these values.
  uint8_t physicalLink = 2;
  uint8_t logicalChannel = 3;
  // The supplier Host program supports both orders; select the deployed FX3
  // firmware's per-32-bit-word representation in the board profile.
  UsbWordByteOrder wordByteOrder = UsbWordByteOrder::BigEndian;
  uint32_t startupDelayMs = 20;
  uint32_t transferTimeoutMs = 5000;
  uint32_t receivePollTimeoutMs = 200;
  uint32_t responseTimeoutMs = 60000;
  size_t maxTransferBytes = kUsbRecordBytes * 256;
  size_t receiveChunkBytes = 32 * 1024;
};

// Linux factory available when the optional libusb transport is built.
std::unique_ptr<CIMDevice> createUsbCIMDevice(const UsbCIMDeviceConfig &config);

} // namespace cim22::runtime

#endif // CIM22_RUNTIME_USBCIMDEVICE_H
