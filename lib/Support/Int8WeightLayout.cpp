//===- Int8WeightLayout.cpp - CIM22 INT8 weight layout --------------------===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIM22/Support/Int8WeightLayout.h"

#include <cstddef>

namespace mlir::cim22 {

std::array<uint32_t, 256>
mapInt8WeightTileToCIMWords(const std::array<uint8_t, 16 * 64> &WeightBytes) {
  std::array<uint32_t, 256> Words{};

  // HWSRC-046 verifies this lane/K-to-address mapping against the final-chip
  // testbench fixtures; CTQ-020 still owns execution validation.
  // FIXME(CTQ-020): Supplier fixture agreement is not board verification.
  for (size_t Q = 0; Q < 32; ++Q) {
    for (size_t R = 0; R < 8; ++R) {
      const size_t Address = 8 * Q + R;
      for (size_t Lane = 0; Lane < 16; ++Lane) {
        const uint32_t UpperKBit =
            (WeightBytes[Lane * 64 + (63 - Q)] >> R) & 1U;
        const uint32_t LowerKBit =
            (WeightBytes[Lane * 64 + (31 - Q)] >> R) & 1U;
        Words[Address] |= UpperKBit << (2 * Lane);
        Words[Address] |= LowerKBit << (2 * Lane + 1);
      }
    }
  }

  return Words;
}

} // namespace mlir::cim22
