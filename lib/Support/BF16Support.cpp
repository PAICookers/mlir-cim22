//===- BF16Support.cpp - CIM22 BF16 numeric support ----------------------===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIM22/Support/BF16Support.h"

#include "llvm/Support/MathExtras.h"

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
    if (sign)
      mantissa = static_cast<uint8_t>(~mantissa + 1U);

    const int exponentDifference =
        static_cast<int>(result.exponent) - (exponent == 0 ? 1 : exponent);
    uint8_t aligned = 0;
    if (exponentDifference < 7) {
      aligned = static_cast<uint8_t>((mantissa >> (exponentDifference + 1)) +
                                     (sign ? 0x80 : 0));
    }
    result.mantissas[index] = static_cast<int8_t>(aligned);
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
  const bool sign = value < 0;
  const uint32_t magnitude = static_cast<uint32_t>(sign ? -value : value);

  int shift = 0;
  uint32_t shifted = 0;
  if (magnitude != 0) {
    shift = static_cast<int>(llvm::Log2_32(magnitude)) - 7;
    shifted = shift >= 0 ? magnitude >> shift : magnitude << -shift;
  }

  const int exponent =
      std::clamp(static_cast<int>(weightExponent) +
                     static_cast<int>(inputExponent) + 1 + shift,
                 0, 255);
  return static_cast<uint16_t>((sign ? 0x8000 : 0) | (exponent << 7) |
                               (shifted & 0x7f));
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
