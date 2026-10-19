//===- BF16SupportTest.cpp - CIM22 BF16 support tests --------------------===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIM22/Support/BF16Support.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>

using namespace mlir::cim22;

static bool check(bool condition, const char *message) {
  if (condition)
    return true;
  std::cerr << "FAIL: " << message << '\n';
  return false;
}

static bool testPrealignmentEdges() {
  std::array<uint16_t, kBF16InputElements> values{};
  values[0] = 0x3f80; // +1.0
  values[1] = 0xbfc0; // -1.5
  values[2] = 0x3f00; // +0.5
  values[3] = 0xbe80; // -0.25
  values[4] = 0x3c00; // Exponent difference 7, truncated.

  const BF16PrealignedVector result = prealignBF16Vector(values);
  return check(result.exponent == 0x7f, "BF16-P01 maximum exponent") &&
         check(result.mantissas[0] == 64, "BF16-P01 positive mantissa") &&
         check(result.mantissas[1] == -96, "BF16-P01 negative mantissa") &&
         check(result.mantissas[2] == 32, "BF16-P01 shifted mantissa") &&
         check(result.mantissas[3] == -16,
               "BF16-P01 magnitude shift before sign") &&
         check(result.mantissas[4] == 0,
               "BF16-P01 exponent-difference truncation");
}

static bool testSupplierPrealignmentFixtures() {
  const std::array<uint16_t, kBF16InputElements> firstInput{
      0x0000, 0x0010, 0x0000, 0x0010, 0x0014, 0x0008, 0x8110, 0x0104,
      0x0208, 0x0411, 0x1400, 0x0000, 0x8406, 0x0000, 0x1042, 0x0081,
      0x0030, 0x0800, 0x0800, 0x0800, 0x0005, 0x0004, 0x0020, 0x0000,
      0x9020, 0x0100, 0x0000, 0x9002, 0x1000, 0x0048, 0x8000, 0x2040,
      0x0000, 0x3000, 0x4104, 0x0080, 0x0280, 0x0000, 0x6081, 0x0008,
      0x0000, 0x7406, 0x8001, 0x0000, 0x0220, 0x0000, 0x0800, 0x4000,
      0x1100, 0x0000, 0x8200, 0x4400, 0x0000, 0x0018, 0x0100, 0x8108,
      0x0518, 0x00a0, 0x0227, 0x0000, 0x1010, 0x1020, 0x4010, 0x0000};
  std::array<int8_t, kBF16InputElements> firstExpected{};
  firstExpected[41] = 0x43;

  const std::array<uint16_t, kBF16InputElements> secondInput{
      0x2020, 0x0000, 0x0004, 0x0698, 0x0000, 0x0040, 0x0a00, 0x0010,
      0x0011, 0x0000, 0x0001, 0x1200, 0x8400, 0x0001, 0x0100, 0x1800,
      0x0004, 0x8808, 0x0000, 0x0402, 0x2101, 0x0808, 0x1080, 0x0020,
      0x0180, 0x0410, 0x0000, 0x2408, 0x90a3, 0x0031, 0x0000, 0x0000,
      0x0820, 0x0000, 0x0010, 0x4040, 0x1004, 0x0001, 0x000a, 0x0000,
      0x0209, 0x4000, 0x0004, 0x0044, 0x0106, 0x0004, 0x0002, 0x8851,
      0x0003, 0x0000, 0x0200, 0x0090, 0x0000, 0x0008, 0x3000, 0x2004,
      0x0030, 0x1020, 0x0028, 0x0001, 0x0020, 0x4000, 0x0020, 0x4020};
  std::array<int8_t, kBF16InputElements> secondExpected{};
  secondExpected[35] = 0x60;
  secondExpected[41] = 0x40;
  secondExpected[61] = 0x40;
  secondExpected[63] = 0x50;

  const BF16PrealignedVector first = prealignBF16Vector(firstInput);
  const BF16PrealignedVector second = prealignBF16Vector(secondInput);
  return check(first.exponent == 0xe8 && first.mantissas == firstExpected,
               "BF16-P02 test_data_1_17 prealignment") &&
         check(second.exponent == 0x80 && second.mantissas == secondExpected,
               "BF16-P02 CIM22_tb prealignment");
}

static bool testInt21Conversion() {
  return check(int21ToBF16(0, 3, 4) == 0x0400,
               "BF16-P03 zero keeps exponent path") &&
         check(int21ToBF16(6144, 127, 127) == 0x7fc0,
               "BF16-P03 positive normalization") &&
         check(int21ToBF16(-6144, 127, 127) == 0xffc0,
               "BF16-P03 negative normalization") &&
         check(int21ToBF16(2048, 1, 127) == 0x4280,
               "BF16-P03 raw exponent reconstruction") &&
         check(int21ToBF16(1048575, 254, 254) == 0x7fff &&
                   int21ToBF16(-1048576, 254, 254) == 0xff80,
               "BF16-P03 exponent clamp, both signs") &&
         check(int21ToBF16(-1048576, 127, 127) == 0xff80,
               "BF16-P03 signed INT21 minimum") &&
         check(int21ToBF16(8191, 127, 127) == 0x7fff &&
                   int21ToBF16(-8191, 127, 127) == 0xffff,
               "BF16-P03 truncate final fraction toward zero") &&
         check(int21ToBF16(1, 0, 0) == 0 && int21ToBF16(-1, 0, 0) == 0x8000,
               "BF16-P03 low clamp preserves sign") &&
         check(int21ToBF16(127, 1, 132) == 0x42fe &&
                   int21ToBF16(128, 1, 132) == 0x4300 &&
                   int21ToBF16(-3, 1, 131) == 0xbfc0,
               "BF16-P03 direct truncation boundaries");
}

static bool testFiniteAlignmentError() {
  // Independent numeric bound across every finite BF16 magnitude and several
  // competing block maxima. Shared-exponent loss must never change the sign or
  // exceed one quantization step; cancellation is exact after alignment.
  for (uint32_t bits = 0; bits < 0x7f80; ++bits) {
    const int exponent = (bits >> 7) & 0xff;
    const double original = std::ldexp((bits & 0x7f) + (exponent ? 128 : 0),
                                       std::max(exponent, 1) - 134);
    for (uint16_t anchor : {0x0000, 0x3f80, 0x7f7f}) {
      std::array<uint16_t, kBF16InputElements> values{};
      values[0] = static_cast<uint16_t>(bits);
      values[1] = static_cast<uint16_t>(bits | 0x8000);
      values[2] = anchor;
      const auto result = prealignBF16Vector(values);
      const double step =
          std::ldexp(1.0, std::max<int>(result.exponent, 1) - 133);
      const double represented = result.mantissas[0] * step;
      if (!check(result.mantissas[0] >= 0 &&
                     result.mantissas[1] == -result.mantissas[0] &&
                     represented <= original && original - represented < step,
                 "BF16-P04 exhaustive finite alignment bound and symmetry"))
        return false;
    }
  }
  return true;
}

static bool testWeightPrealignment() {
  std::array<uint16_t, kBF16OutputElements * kBF16InputElements> weights{};
  for (size_t lane = 0; lane < kBF16OutputElements; ++lane) {
    weights[lane * kBF16InputElements] =
        static_cast<uint16_t>(0x3f80 + 0x80 * lane);
    weights[lane * kBF16InputElements + 1] =
        static_cast<uint16_t>(0xbf80 + 0x80 * lane);
  }
  const BF16PrealignedWeightTile result = prealignBF16WeightTile(weights);
  for (size_t lane = 0; lane < kBF16OutputElements; ++lane) {
    if (!check(result.exponents[lane] == 0x7f + lane,
               "BF16-P05 per-lane weight exponent") ||
        !check(result.mantissas[lane * kBF16InputElements] == 64,
               "BF16-P05 positive weight mantissa") ||
        !check(result.mantissas[lane * kBF16InputElements + 1] == -64,
               "BF16-P05 negative weight mantissa"))
      return false;
  }
  return true;
}

static bool testZeroAndSubnormalBits() {
  std::array<uint16_t, kBF16InputElements> values{};
  values[1] = 0x8000;
  values[2] = 0x0002;
  values[3] = 0x8002;
  values[4] = 0x0001; // Lost to the one-bit headroom rule.
  const BF16PrealignedVector aligned = prealignBF16Vector(values);
  return check(aligned.exponent == 0 && aligned.mantissas[0] == 0 &&
                   aligned.mantissas[1] == 0 && aligned.mantissas[2] == 1 &&
                   aligned.mantissas[3] == -1 && aligned.mantissas[4] == 0,
               "BF16-P07 zero/subnormal prealignment") &&
         check(int21ToBF16(262144, 127, 127) == 0x7f80,
               "BF16-P07 64 unit products") &&
         check(int21ToBF16(64, 0, 127) == 0x3f80,
               "BF16-P07 raw exponent uses direct scale");
}

static bool testExponentGapBoundary() {
  std::array<uint16_t, kBF16InputElements> values{};
  values[0] = 0x3f80;
  values[1] = 0x3c80; // Gap 6: keep the last positive bit.
  values[2] = 0xbc80; // Negative magnitude also retains one bit.
  values[3] = 0x3c00; // Gap 7: drop both signs completely.
  values[4] = 0xbc00;
  values[5] = 0x3b80; // Gap 8: also zero.
  values[6] = 0xbb80;
  values[7] = 0xbf02; // -0.5078125: truncate magnitude, not signed floor.
  const auto aligned = prealignBF16Vector(values);
  return check(aligned.mantissas[1] == 1 && aligned.mantissas[2] == -1 &&
                   aligned.mantissas[3] == 0 && aligned.mantissas[4] == 0 &&
                   aligned.mantissas[5] == 0 && aligned.mantissas[6] == 0 &&
                   aligned.mantissas[7] == -32,
               "BF16-P08 signed exponent gaps and toward-zero shift");
}

static bool testLayouts() {
  std::array<uint8_t, kBF16OutputElements> exponents{};
  for (size_t lane = 0; lane < exponents.size(); ++lane)
    exponents[lane] = static_cast<uint8_t>(lane);
  const auto exponentWords = packBF16WeightExponents(exponents);
  if (!check(exponentWords[0] == 0x0706050403020100ULL &&
                 exponentWords[1] == 0x0f0e0d0c0b0a0908ULL,
             "BF16-P06 Weight_EXP little-endian lanes"))
    return false;

  std::array<uint16_t, kBF16InputElements> input{};
  for (size_t element = 0; element < input.size(); ++element)
    input[element] = static_cast<uint16_t>(0x100 * element + element);
  const auto inputWords = packBF16InputCacheRow(input);
  if (!check(inputWords[0] == 0x0000010102020303ULL &&
                 inputWords[15] == 0x3c3c3d3d3e3e3f3fULL,
             "BF16-P06 input Cache body order"))
    return false;

  input[0] = 0x3f80;
  input[1] = 0xbf00;
  input[2] = 0x0001;
  input[3] = 0x8000;
  if (!check(packBF16InputCacheRow(input)[0] == 0x3f80bf0000018000ULL,
             "BF16-P06 preserve unequal element bytes and signed zero"))
    return false;

  // Independent fixture: concatenate lane 15..0 as 21-bit text, then 48
  // padding bits; split into six 64-bit words. No decoder offset formula.
  const std::array<uint16_t, kBF16OutputElements> expected{
      0x0000, 0x8000, 0x0001, 0x8001, 0x3f80, 0xbf80, 0x7f7f, 0xff7f,
      0x7f80, 0xff80, 0x7fc1, 0xffff, 0x1234, 0xabcd, 0x55aa, 0xaa55};
  const std::array<uint64_t, 6> response{
      0x0552a8156a81579aULL, 0x0123407fff81ff04ULL, 0x1ff0007f8007fbf8ULL,
      0x1fdfc17f0003f800ULL, 0x4000800004100000ULL, 0x0000000000000000ULL};
  // Set the five unused bits of every lane and all 48 trailing bits. They
  // must not enter the returned BF16 payload (not a claim of legal RTL data).
  const std::array<uint64_t, 6> noisyResponse{
      0xfd52afd56abf579bULL, 0xf1234ffffffdff07ULL, 0xfff01f7f80fffbffULL,
      0xdfdfff7f01f3f80fULL, 0xc000fc0007f0001fULL, 0x0000ffffffffffffULL};
  return check(decodeBF16OutputCacheResponse(response) == expected,
               "BF16-P06 output Cache fixed 336+48 decode") &&
         check(decodeBF16OutputCacheResponse(noisyResponse) == expected,
               "BF16-P06 ignore lane upper bits and trailing padding");
}

int main() {
  return testPrealignmentEdges() && testSupplierPrealignmentFixtures() &&
                 testInt21Conversion() && testFiniteAlignmentError() &&
                 testWeightPrealignment() && testZeroAndSubnormalBits() &&
                 testExponentGapBoundary() && testLayouts()
             ? 0
             : 1;
}
