//===- Int8WeightLayoutTest.cpp - INT8 weight layout tests ---------------===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIM22/Support/Int8WeightLayout.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>

constexpr std::size_t NumLanes = 16;
constexpr std::size_t WeightsPerLane = 64;
constexpr std::size_t NumWords = 256;

using WeightTile = std::array<uint8_t, NumLanes * WeightsPerLane>;
using CIMWords = std::array<uint32_t, NumWords>;

static CIMWords mapReference(const WeightTile &WeightBytes) {
  CIMWords Words{};
  for (std::size_t Lane = 0; Lane < NumLanes; ++Lane) {
    for (std::size_t K = 0; K < WeightsPerLane; ++K) {
      const bool IsUpperHalf = K >= 32;
      const std::size_t Q = IsUpperHalf ? 63 - K : 31 - K;
      const std::size_t WordBit = 2 * Lane + (IsUpperHalf ? 0 : 1);
      for (std::size_t ByteBit = 0; ByteBit < 8; ++ByteBit) {
        const uint32_t ValueBit =
            (WeightBytes[Lane * WeightsPerLane + K] >> ByteBit) & 1U;
        Words[8 * Q + ByteBit] |= ValueBit << WordBit;
      }
    }
  }
  return Words;
}

static WeightTile invertReference(const CIMWords &Words) {
  WeightTile WeightBytes{};
  for (std::size_t Lane = 0; Lane < NumLanes; ++Lane) {
    for (std::size_t K = 0; K < WeightsPerLane; ++K) {
      const bool IsUpperHalf = K >= 32;
      const std::size_t Q = IsUpperHalf ? 63 - K : 31 - K;
      const std::size_t WordBit = 2 * Lane + (IsUpperHalf ? 0 : 1);
      for (std::size_t R = 0; R < 8; ++R)
        WeightBytes[Lane * WeightsPerLane + K] |=
            static_cast<uint8_t>(((Words[8 * Q + R] >> WordBit) & 1U) << R);
    }
  }
  return WeightBytes;
}

static bool check(bool Condition, const char *Message) {
  if (Condition)
    return true;
  std::cerr << "FAIL: " << Message << '\n';
  return false;
}

static bool testAllZero() {
  const WeightTile WeightBytes{};
  const CIMWords Words = mlir::cim22::mapInt8WeightTileToCIMWords(WeightBytes);
  return check(Words == CIMWords{}, "WL-P01 all-zero tile");
}

static bool testAllOnes() {
  WeightTile WeightBytes{};
  WeightBytes.fill(0xff);
  CIMWords Expected{};
  Expected.fill(std::numeric_limits<uint32_t>::max());
  const CIMWords Words = mlir::cim22::mapInt8WeightTileToCIMWords(WeightBytes);
  return check(Words == Expected, "WL-P02 all-one tile");
}

static bool testOneHotBoundaries() {
  constexpr std::array<std::size_t, 2> Lanes{0, 15};
  constexpr std::array<std::size_t, 4> KValues{0, 31, 32, 63};
  constexpr std::array<std::size_t, 2> ByteBits{0, 7};

  for (const std::size_t Lane : Lanes) {
    for (const std::size_t K : KValues) {
      for (const std::size_t ByteBit : ByteBits) {
        WeightTile WeightBytes{};
        WeightBytes[Lane * WeightsPerLane + K] = uint8_t{1} << ByteBit;

        CIMWords Expected{};
        const bool IsUpperHalf = K >= 32;
        const std::size_t Q = IsUpperHalf ? 63 - K : 31 - K;
        const std::size_t WordBit = 2 * Lane + (IsUpperHalf ? 0 : 1);
        Expected[8 * Q + ByteBit] = uint32_t{1} << WordBit;

        const CIMWords Words =
            mlir::cim22::mapInt8WeightTileToCIMWords(WeightBytes);
        if (!check(Words == Expected, "WL-P03 one-hot boundary"))
          return false;
      }
    }
  }
  return true;
}

static bool testAsymmetricRoundTrip() {
  WeightTile WeightBytes{};
  for (std::size_t Lane = 0; Lane < NumLanes; ++Lane) {
    for (std::size_t K = 0; K < WeightsPerLane; ++K) {
      WeightBytes[Lane * WeightsPerLane + K] =
          static_cast<uint8_t>(0x5aU + 67U * Lane + 29U * K + 7U * (Lane ^ K));
    }
  }

  const CIMWords Expected = mapReference(WeightBytes);
  const CIMWords First = mlir::cim22::mapInt8WeightTileToCIMWords(WeightBytes);
  const CIMWords Second = mlir::cim22::mapInt8WeightTileToCIMWords(WeightBytes);
  const WeightTile Inverse = mlir::cim22::unmapCIMWordsToInt8WeightTile(First);
  return check(First == Expected, "WL-P04 independent reference") &&
         check(invertReference(First) == WeightBytes,
               "WL-P04 independent inverse") &&
         check(Inverse == WeightBytes, "WL-P04 production inverse") &&
         check(Second == First, "WL-P04 deterministic mapping");
}

int main() {
  return testAllZero() && testAllOnes() && testOneHotBoundaries() &&
                 testAsymmetricRoundTrip()
             ? 0
             : 1;
}
