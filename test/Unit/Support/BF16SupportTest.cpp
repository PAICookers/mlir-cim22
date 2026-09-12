//===- BF16SupportTest.cpp - CIM22 BF16 support tests --------------------===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIM22/Support/BF16Support.h"

#include <array>
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
         check(static_cast<uint8_t>(result.mantissas[3]) == 0x90,
               "BF16-P01 hardware negative shift") &&
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
               "BF16-P03 zero follows hardware exponent path") &&
         check(int21ToBF16(192, 0, 126) == 0x3fc0,
               "BF16-P03 positive normalization") &&
         check(int21ToBF16(-192, 0, 126) == 0xbfc0,
               "BF16-P03 negative normalization") &&
         check(int21ToBF16(3, 0, 0) == 0x0040,
               "BF16-P03 exponent underflow saturation") &&
         check(int21ToBF16(1048575, 254, 254) == 0x7fff,
               "BF16-P03 exponent overflow saturation") &&
         check(int21ToBF16(-1048576, 0, 0) == 0x8700,
               "BF16-P03 signed INT21 minimum");
}

static bool testSupplierInt21Fixture() {
  const std::array<uint8_t, kBF16OutputElements> weightExponents{
      32, 4, 16, 2, 128, 8, 1, 4, 64, 8, 2, 1, 4, 128, 2, 64};
  const std::array<int32_t, kBF16OutputElements> values{
      -11648, 160,  -1216, 6864,  -12288, 640,    -4864, 512,
      6400,   2048, 5504,  -7584, -3328,  -11840, 1344,  0};
  const std::array<uint16_t, kBF16OutputElements> expected{
      0xd3b6, 0x42a0, 0xca18, 0x4456, 0xffc0, 0x45a0, 0xc398, 0x4380,
      0x6348, 0x4680, 0x442c, 0xc3ed, 0xc4d0, 0xffb9, 0x4328, 0x6080};

  std::array<uint16_t, kBF16OutputElements> actual{};
  for (size_t lane = 0; lane < actual.size(); ++lane)
    actual[lane] = int21ToBF16(values[lane], weightExponents[lane], 128);
  return check(actual == expected, "BF16-P04 supplier INT21 conversion");
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

static bool testSupplierZeroAndSubnormalBits() {
  // Freeze the supplier bit operations, not IEEE zero/subnormal semantics.
  std::array<uint16_t, kBF16InputElements> values{};
  values[1] = 0x8000;
  values[2] = 0x0001;
  values[3] = 0x8001;
  const BF16PrealignedVector aligned = prealignBF16Vector(values);
  return check(aligned.exponent == 0 && aligned.mantissas[0] == 0 &&
                   aligned.mantissas[1] == -128 && aligned.mantissas[2] == 1 &&
                   aligned.mantissas[3] == 127,
               "BF16-P07 supplier zero/subnormal prealignment") &&
         check(int21ToBF16(262144, 127, 127) == 0x7f80,
               "BF16-P07 raw unit weights are not IEEE arithmetic") &&
         check(int21ToBF16(0, 0, 127) == 0x4000,
               "BF16-P07 zero dot retains the supplier exponent");
}

static bool testExponentGapBoundary() {
  std::array<uint16_t, kBF16InputElements> values{};
  values[0] = 0x3f80;
  values[1] = 0x3c80; // Gap 6: keep the last positive bit.
  values[2] = 0xbc80; // Negative sign extension survives gap 6.
  values[3] = 0x3c00; // Gap 7: drop both signs completely.
  values[4] = 0xbc00;
  values[5] = 0x3b80; // Gap 8: also zero.
  values[6] = 0xbb80;
  const auto aligned = prealignBF16Vector(values);
  return check(aligned.mantissas[1] == 1 && aligned.mantissas[2] == -127 &&
                   aligned.mantissas[3] == 0 && aligned.mantissas[4] == 0 &&
                   aligned.mantissas[5] == 0 && aligned.mantissas[6] == 0,
               "BF16-P08 signed exponent gaps 6/7/8") &&
         check(int21ToBF16(255, 0, 253) == 0x7f7f &&
                   int21ToBF16(255, 0, 254) == 0x7fff &&
                   int21ToBF16(255, 1, 254) == 0x7fff,
               "BF16-P08 exponent 254/255/256 retains fraction") &&
         check(int21ToBF16(3, 0, 4) == 0x0040 &&
                   int21ToBF16(3, 0, 5) == 0x0040 &&
                   int21ToBF16(3, 0, 6) == 0x00c0,
               "BF16-P08 exponent -1/0/1 retains fraction");
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
                 testInt21Conversion() && testSupplierInt21Fixture() &&
                 testWeightPrealignment() &&
                 testSupplierZeroAndSubnormalBits() &&
                 testExponentGapBoundary() && testLayouts()
             ? 0
             : 1;
}
