//===- UsbRecordCodec.h - CIM22 FPGA USB records -------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIM22_RUNTIME_USBRECORDCODEC_H
#define CIM22_RUNTIME_USBRECORDCODEC_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace cim22::runtime {

inline constexpr size_t kUsbRecordBytes = 12;

struct UsbRecord {
  uint8_t physicalLink = 0;
  uint8_t logicalChannel = 0;
  uint64_t flit = 0;
};

enum class UsbWordByteOrder : uint8_t {
  BigEndian,
  LittleEndian,
};

class UsbRecordCodec final {
public:
  explicit UsbRecordCodec(
      UsbWordByteOrder byteOrder = UsbWordByteOrder::BigEndian)
      : byteOrder_(byteOrder) {}

  static llvm::Error validate(const UsbRecord &record);
  static std::array<uint8_t, kUsbRecordBytes>
  encode(const UsbRecord &record,
         UsbWordByteOrder byteOrder = UsbWordByteOrder::BigEndian);

  // Appends complete records to `records`; at most 11 residual bytes remain.
  // On a malformed record the stream must be discarded.
  llvm::Error append(llvm::ArrayRef<uint8_t> bytes,
                     llvm::SmallVectorImpl<UsbRecord> &records);

  size_t pendingBytes() const { return pending_.size(); }

private:
  UsbWordByteOrder byteOrder_;
  llvm::SmallVector<uint8_t, kUsbRecordBytes> pending_;
};

} // namespace cim22::runtime

#endif // CIM22_RUNTIME_USBRECORDCODEC_H
