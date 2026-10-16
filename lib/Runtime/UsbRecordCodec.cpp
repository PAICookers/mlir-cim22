//===- UsbRecordCodec.cpp - CIM22 FPGA USB records -----------*- C++ -*-===//

#include "CIM22/Runtime/UsbRecordCodec.h"

#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"

#include <cstddef>

namespace cim22::runtime {
namespace {
llvm::Error codecError(const char *message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

uint32_t loadWord(llvm::ArrayRef<uint8_t> bytes, size_t offset,
                  UsbWordByteOrder byteOrder) {
  const void *word = bytes.data() + offset;
  return byteOrder == UsbWordByteOrder::LittleEndian
             ? llvm::support::endian::read32le(word)
             : llvm::support::endian::read32be(word);
}

void storeWord(uint32_t value, llvm::MutableArrayRef<uint8_t> bytes,
               size_t offset, UsbWordByteOrder byteOrder) {
  void *word = bytes.data() + offset;
  if (byteOrder == UsbWordByteOrder::LittleEndian)
    llvm::support::endian::write32le(word, value);
  else
    llvm::support::endian::write32be(word, value);
}
} // namespace

std::array<uint8_t, kUsbRecordBytes>
UsbRecordCodec::encode(const UsbRecord &record, UsbWordByteOrder byteOrder) {
  std::array<uint8_t, kUsbRecordBytes> bytes{};
  const uint32_t header =
      (static_cast<uint32_t>(record.physicalLink & 0x3) << 3) |
      static_cast<uint32_t>(record.logicalChannel & 0x7);
  storeWord(header, llvm::MutableArrayRef<uint8_t>(bytes), 0, byteOrder);
  storeWord(static_cast<uint32_t>(record.flit >> 32),
            llvm::MutableArrayRef<uint8_t>(bytes), 4, byteOrder);
  storeWord(static_cast<uint32_t>(record.flit),
            llvm::MutableArrayRef<uint8_t>(bytes), 8, byteOrder);
  return bytes;
}

llvm::Error UsbRecordCodec::validate(const UsbRecord &record) {
  if (record.physicalLink > 3)
    return codecError("CIM USB record physical link is outside [0, 3]");
  if (record.logicalChannel > 7)
    return codecError("CIM USB record logical channel is outside [0, 7]");
  return llvm::Error::success();
}

llvm::Error UsbRecordCodec::append(llvm::ArrayRef<uint8_t> bytes,
                                   llvm::SmallVectorImpl<UsbRecord> &records) {
  pending_.append(bytes.begin(), bytes.end());
  size_t offset = 0;
  while (pending_.size() - offset >= kUsbRecordBytes) {
    const auto record =
        llvm::ArrayRef<uint8_t>(pending_).slice(offset, kUsbRecordBytes);
    const uint32_t header = loadWord(record, 0, byteOrder_);
    if ((header & ~uint32_t{0x1f}) != 0)
      return codecError("CIM USB response has non-zero reserved header bits");
    records.push_back(
        {static_cast<uint8_t>((header >> 3) & 0x3),
         static_cast<uint8_t>(header & 0x7),
         (static_cast<uint64_t>(loadWord(record, 4, byteOrder_)) << 32) |
             loadWord(record, 8, byteOrder_)});
    offset += kUsbRecordBytes;
  }
  pending_.erase(pending_.begin(), pending_.begin() + offset);
  return llvm::Error::success();
}

} // namespace cim22::runtime
