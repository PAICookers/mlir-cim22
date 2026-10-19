//===- BF16Support.cpp - CIM22 BF16 numeric support ----------------------===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIM22/Support/BF16Support.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace mlir::cim22 {

BF16PrealignedVector
prealignBF16Vector(const std::array<uint16_t, kBF16InputElements> &values) {
  BF16PrealignedVector result;
  for (uint16_t value : values)
    result.exponent =
        std::max(result.exponent, static_cast<uint8_t>((value >> 7) & 0xff));

  for (size_t index = 0; index < values.size(); ++index) {
    const uint16_t value = values[index];
    const bool sign = (value >> 15) != 0;
    const uint8_t exponent = static_cast<uint8_t>((value >> 7) & 0xff);
    uint8_t mantissa = static_cast<uint8_t>(value & 0x7f);
    if (exponent != 0)
      mantissa |= 0x80;
    // Reserve one headroom bit. Truncate magnitude before applying the sign,
    // so positive and negative values use the same scale and rounding rule.
    // Keep raw exponent bytes in Weight_EXP; subnormals use effective E=1.
    const int shift =
        std::max<int>(result.exponent, 1) - std::max<int>(exponent, 1) + 1;
    const int aligned = shift >= 8 ? 0 : mantissa >> shift;
    result.mantissas[index] = static_cast<int8_t>(sign ? -aligned : aligned);
  }
  return result;
}

BF16PrealignedWeightTile prealignBF16WeightTile(
    const std::array<uint16_t, kBF16OutputElements * kBF16InputElements>
        &values) {
  BF16PrealignedWeightTile result;
  for (size_t lane = 0; lane < kBF16OutputElements; ++lane) {
    std::array<uint16_t, kBF16InputElements> laneValues{};
    std::copy_n(values.begin() + lane * kBF16InputElements, kBF16InputElements,
                laneValues.begin());
    const BF16PrealignedVector laneResult = prealignBF16Vector(laneValues);
    result.exponents[lane] = laneResult.exponent;
    std::copy(laneResult.mantissas.begin(), laneResult.mantissas.end(),
              result.mantissas.begin() + lane * kBF16InputElements);
  }
  return result;
}

uint16_t int21ToBF16(int32_t value, uint8_t weightExponent,
                     uint8_t inputExponent) {
  // Match the RTL-derived model: retain the highest eight magnitude bits,
  // truncate toward zero, then combine the raw exponent sum and sign.
  const bool negative = value < 0;
  const int64_t magnitude = negative ? -static_cast<int64_t>(value) : value;
  int shift = 0;
  uint64_t mantissaShifted = 0;
  if (magnitude != 0) {
    int highestBit = 0;
    for (int64_t remaining = magnitude; remaining > 1; remaining >>= 1)
      ++highestBit;
    shift = highestBit - 7;
    mantissaShifted = shift >= 0 ? static_cast<uint64_t>(magnitude) >> shift
                                 : static_cast<uint64_t>(magnitude) << -shift;
  }

  const int exponent = static_cast<int>(weightExponent) +
                       static_cast<int>(inputExponent) + 1 + shift;
  const uint16_t exponent8 =
      static_cast<uint16_t>(std::clamp(exponent, 0, 255));
  const uint16_t sign = negative ? 0x8000 : 0;
  return static_cast<uint16_t>(sign | (exponent8 << 7) |
                               (mantissaShifted & 0x7f));
}

std::array<uint64_t, 2> packBF16WeightExponents(
    const std::array<uint8_t, kBF16OutputElements> &exponents) {
  std::array<uint64_t, 2> words{};
  for (size_t index = 0; index < exponents.size(); ++index)
    words[index / 8] |= static_cast<uint64_t>(exponents[index])
                        << (8 * (index % 8));
  return words;
}

std::array<uint64_t, 16>
packBF16InputCacheRow(const std::array<uint16_t, kBF16InputElements> &values) {
  std::array<uint64_t, 16> words{};
  for (size_t word = 0; word < words.size(); ++word)
    for (size_t element = 0; element < 4; ++element)
      words[word] = (words[word] << 16) | values[word * 4 + element];
  return words;
}

std::array<uint16_t, kBF16OutputElements>
decodeBF16OutputCacheResponse(const std::array<uint64_t, 6> &responseFlits) {
  std::array<uint16_t, kBF16OutputElements> values{};
  for (size_t lane = 0; lane < values.size(); ++lane) {
    const size_t bitOffset = 48 + 21 * lane;
    const size_t flit = 5 - bitOffset / 64;
    values[lane] = static_cast<uint16_t>(
        (responseFlits[flit] >> (bitOffset % 64)) & 0xffff);
  }
  return values;
}

} // namespace mlir::cim22
