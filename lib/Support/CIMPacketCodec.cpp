//===- CIMPacketCodec.cpp - Transport-neutral CIM22 flits -----*- C++ -*-===//

#include "CIM22/Runtime/CIMPacketCodec.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>

namespace cim22::runtime {
namespace {
constexpr unsigned kTypeShift = 60;
constexpr std::array<unsigned, 6> kRouteShifts{54, 48, 42, 36, 30, 24};
constexpr uint64_t kControlType = uint64_t{0x8} << kTypeShift;
constexpr uint64_t kWorkOnceType = uint64_t{0x9} << kTypeShift;
constexpr uint64_t kInputType = uint64_t{0x1} << kTypeShift;
constexpr uint64_t kReturnRouteType = uint64_t{0xa} << kTypeShift;
constexpr uint64_t kCacheReadType = uint64_t{0x5} << kTypeShift;
constexpr uint64_t kCIMWriteType = uint64_t{0x2} << kTypeShift;
constexpr uint64_t kWeightExponentType = uint64_t{0x3} << kTypeShift;
constexpr uint64_t kRequestBit = uint64_t{1} << 23;

llvm::Error codecError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

uint64_t encodeSignedMagnitude6(int32_t value) {
  const uint64_t magnitude = static_cast<uint64_t>(value < 0 ? -value : value);
  return value < 0 ? magnitude | uint64_t{0x20} : magnitude;
}

uint64_t encodeRoute(llvm::ArrayRef<int32_t> route) {
  uint64_t encoded = 0;
  for (auto [value, shift] : llvm::zip_equal(route, kRouteShifts))
    encoded |= encodeSignedMagnitude6(value) << shift;
  return encoded;
}

llvm::Error validateRoute(llvm::ArrayRef<int32_t> route) {
  if (route.size() != 6)
    return codecError("CIM route requires six fields");
  for (auto [index, value] : llvm::enumerate(route)) {
    if (value < -31 || value > 31)
      return codecError("CIM route field is outside [-31, 31]");
    if (index >= 3 && value != 0)
      return codecError("CIM runtime requires zero Copy route fields");
  }
  return llvm::Error::success();
}

int32_t signExtend21(uint32_t value) {
  return static_cast<int32_t>(value << 11) >> 11;
}
} // namespace

llvm::Error validateCIMRoute(llvm::ArrayRef<int32_t> route) {
  return validateRoute(route);
}

uint64_t encodeCIMControlFlit(llvm::ArrayRef<int32_t> route, int64_t macroSlot,
                              execution::CIMDataType dataType) {
  const uint64_t mode = dataType == execution::CIMDataType::BF16 ? 2 : 0;
  return kControlType | encodeRoute(route) | mode |
         static_cast<uint64_t>(macroSlot);
}

uint64_t encodeCIMWorkOnceFlit(llvm::ArrayRef<int32_t> route) {
  return kWorkOnceType | encodeRoute(route) | 1;
}

uint64_t encodeCIMWorkIdleFlit(llvm::ArrayRef<int32_t> route) {
  return kWorkOnceType | encodeRoute(route);
}

uint64_t encodeCIMReturnRouteFlit(llvm::ArrayRef<int32_t> route,
                                  int64_t coreSlot,
                                  llvm::ArrayRef<int32_t> testCore) {
  return kReturnRouteType | encodeRoute(route) |
         (static_cast<uint64_t>(coreSlot) << 18) |
         (encodeSignedMagnitude6(testCore[0]) << 12) |
         (encodeSignedMagnitude6(testCore[1]) << 6) |
         encodeSignedMagnitude6(testCore[2]);
}

uint64_t encodeCIMOutputCacheReadFlit(llvm::ArrayRef<int32_t> route,
                                      int64_t cacheAddress) {
  return kCacheReadType | encodeRoute(route) | kRequestBit |
         (static_cast<uint64_t>(cacheAddress) << 14);
}

uint64_t encodeCIMWeightPayload(uint32_t word, uint8_t address) {
  return (static_cast<uint64_t>(word) << 8) | address;
}

llvm::Error appendCIMInputCachePacket(llvm::ArrayRef<int32_t> route,
                                      int64_t cacheAddress,
                                      llvm::ArrayRef<uint64_t> words,
                                      llvm::SmallVectorImpl<uint64_t> &flits) {
  if (llvm::Error error = validateRoute(route))
    return error;
  if (cacheAddress < 0 || cacheAddress >= 8)
    return codecError("CIM input Cache address is outside [0, 7]");
  if (words.size() != 16)
    return codecError("CIM input Cache packet requires 16 body flits");
  flits.push_back(
      kInputType | encodeRoute(route) |
      ((uint64_t{0x8} | static_cast<uint64_t>(cacheAddress)) << 14) | 16);
  flits.append(words.begin(), words.end());
  return llvm::Error::success();
}

llvm::Error
appendCIMWeightExponentPacket(llvm::ArrayRef<int32_t> route,
                              llvm::ArrayRef<uint64_t> exponentWords,
                              llvm::SmallVectorImpl<uint64_t> &flits) {
  if (llvm::Error error = validateRoute(route))
    return error;
  if (exponentWords.size() != 2)
    return codecError("CIM Weight_EXP packet requires two body flits");
  flits.push_back(kWeightExponentType | encodeRoute(route) | 2);
  flits.append(exponentWords.begin(), exponentWords.end());
  return llvm::Error::success();
}

llvm::Error appendCIMWeightWordsPacket(llvm::ArrayRef<int32_t> route,
                                       llvm::ArrayRef<uint32_t> words,
                                       llvm::SmallVectorImpl<uint64_t> &flits) {
  if (llvm::Error error = validateRoute(route))
    return error;
  if (words.size() != 256)
    return codecError("CIM weight packet requires 256 body words");
  flits.push_back(kCIMWriteType | encodeRoute(route) | 256);
  for (auto [address, word] : llvm::enumerate(words))
    flits.push_back(
        encodeCIMWeightPayload(word, static_cast<uint8_t>(address)));
  return llvm::Error::success();
}

std::array<uint64_t, 16>
packCIMInt8InputCacheRow(llvm::ArrayRef<int8_t> values) {
  std::array<uint64_t, 16> words{};
  if (values.size() != 64)
    return words;
  // Match the logical K order of the weight layout and supplier Cache rows.
  for (size_t lane = 0; lane < values.size(); ++lane) {
    const size_t word = lane / 4;
    const size_t shift = 16 * (3 - lane % 4);
    words[word] |= static_cast<uint64_t>(static_cast<uint8_t>(values[lane]))
                   << shift;
  }
  return words;
}

llvm::Expected<std::array<int32_t, 16>> decodeCIMInt21OutputCacheResponse(
    const std::array<uint64_t, 6> &responseFlits) {
  if ((responseFlits[5] & uint64_t{0x0000ffffffffffff}) != 0)
    return codecError("CIM output Cache response has non-zero padding");

  std::array<int32_t, 16> values{};
  for (size_t lane = 0; lane < values.size(); ++lane) {
    const size_t bitOffset = 48 + 21 * lane;
    const size_t firstWord = 5 - bitOffset / 64;
    const size_t shift = bitOffset % 64;
    uint64_t bits = responseFlits[firstWord] >> shift;
    if (shift > 43)
      bits |= responseFlits[firstWord - 1] << (64 - shift);
    values[lane] = signExtend21(static_cast<uint32_t>(bits & 0x1fffff));
  }
  return values;
}

} // namespace cim22::runtime
