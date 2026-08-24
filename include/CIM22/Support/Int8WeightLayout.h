//===- Int8WeightLayout.h - CIM22 INT8 weight layout -----------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIM22_SUPPORT_INT8WEIGHTLAYOUT_H
#define CIM22_SUPPORT_INT8WEIGHTLAYOUT_H

#include <array>
#include <cstdint>

namespace mlir::cim22 {

std::array<uint32_t, 256>
mapInt8WeightTileToCIMWords(const std::array<uint8_t, 16 * 64> &WeightBytes);

std::array<uint8_t, 16 * 64>
unmapCIMWordsToInt8WeightTile(const std::array<uint32_t, 256> &Words);

} // namespace mlir::cim22

#endif // CIM22_SUPPORT_INT8WEIGHTLAYOUT_H
