//===- BF16Support.h - CIM22 BF16 numeric support ------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIM22_SUPPORT_BF16SUPPORT_H
#define CIM22_SUPPORT_BF16SUPPORT_H

#include <array>
#include <cstddef>
#include <cstdint>

namespace mlir::cim22 {

inline constexpr std::size_t kBF16InputElements = 64;
inline constexpr std::size_t kBF16OutputElements = 16;

struct BF16PrealignedVector {
  uint8_t exponent = 0;
  std::array<int8_t, kBF16InputElements> mantissas{};
};

struct BF16PrealignedWeightTile {
  std::array<int8_t, kBF16OutputElements * kBF16InputElements> mantissas{};
  std::array<uint8_t, kBF16OutputElements> exponents{};
};

// Reproduces the CIM22 input exponent-prealignment datapath on BF16 bit
// patterns. IEEE special-value interpretation is outside this bit-level API.
BF16PrealignedVector
prealignBF16Vector(const std::array<uint16_t, kBF16InputElements> &values);

// Prealigns each of the 16 logical weight lanes independently.
BF16PrealignedWeightTile prealignBF16WeightTile(
    const std::array<uint16_t, kBF16OutputElements * kBF16InputElements>
        &values);

// Reconstructs the supplier-model BF16 bit pattern from a signed INT21 result.
// Exponent underflow and overflow clamp to 0 and 255; the fraction is retained.
// Zero follows the exponent path too. This is not an IEEE numeric conversion.
uint16_t int21ToBF16(int32_t value, uint8_t weightExponent,
                     uint8_t inputExponent);

// Packs lane 0 in bits [7:0] and lane 15 in bits [63:56] of the second word.
std::array<uint64_t, 2> packBF16WeightExponents(
    const std::array<uint8_t, kBF16OutputElements> &exponents);

// Packs each group of four logical BF16 elements in the high-to-low order used
// by the supplier Cache-write body stream.
std::array<uint64_t, 16>
packBF16InputCacheRow(const std::array<uint16_t, kBF16InputElements> &values);

// Decodes the low 16 bits of each 21-bit output lane from a 336+48-bit Cache
// response. Response flits are supplied in transmitted MSB-to-LSB order.
std::array<uint16_t, kBF16OutputElements>
decodeBF16OutputCacheResponse(const std::array<uint64_t, 6> &responseFlits);

} // namespace mlir::cim22

#endif // CIM22_SUPPORT_BF16SUPPORT_H
