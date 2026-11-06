//===- CIMPacketCodec.h - Transport-neutral CIM22 flits -------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIM22_RUNTIME_CIMPACKETCODEC_H
#define CIM22_RUNTIME_CIMPACKETCODEC_H

#include "CIM22/Execution/CIMExecutable.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <array>
#include <cstdint>

namespace cim22::runtime {

// Validate six signed route fields for onecast (all Copy fields zero).
llvm::Error validateCIMRoute(llvm::ArrayRef<int32_t> route);

// Header encoders expect a legal route. Callers may use the target's
// full-array broadcast route for readback selection and Cache requests.
uint64_t encodeCIMControlFlit(llvm::ArrayRef<int32_t> route, int64_t macroSlot,
                              execution::CIMDataType dataType);
uint64_t encodeCIMWorkOnceFlit(llvm::ArrayRef<int32_t> route);
uint64_t encodeCIMWorkIdleFlit(llvm::ArrayRef<int32_t> route);
uint64_t encodeCIMReturnRouteFlit(llvm::ArrayRef<int32_t> route,
                                  int64_t coreSlot,
                                  llvm::ArrayRef<int32_t> testCore);
uint64_t encodeCIMOutputCacheReadFlit(llvm::ArrayRef<int32_t> route,
                                      int64_t cacheAddress);
uint64_t encodeCIMWeightPayload(uint32_t word, uint8_t address);

// Identifies a response within a single Macro batch.
struct CIMOutputCacheResponseKey {
  uint8_t coreIndex;
  uint8_t cacheAddress;
};

// Validate the reply type and six-flit payload length, then extract Core
// idx[22:18] and Cache row[17:14]. Macro identity is implicit in selection.
llvm::Expected<CIMOutputCacheResponseKey>
decodeCIMOutputCacheResponseHeader(uint64_t header);

// Append a onecast input packet with 16 body flits for one Cache row [0, 7].
llvm::Error appendCIMInputCachePacket(llvm::ArrayRef<int32_t> route,
                                      int64_t cacheAddress,
                                      llvm::ArrayRef<uint64_t> words,
                                      llvm::SmallVectorImpl<uint64_t> &flits);
// Append a onecast Weight_EXP packet with exactly two body flits.
llvm::Error
appendCIMWeightExponentPacket(llvm::ArrayRef<int32_t> route,
                              llvm::ArrayRef<uint64_t> exponentWords,
                              llvm::SmallVectorImpl<uint64_t> &flits);
// Append a onecast Weight packet from 256 words in physical address order.
llvm::Error appendCIMWeightWordsPacket(llvm::ArrayRef<int32_t> route,
                                       llvm::ArrayRef<uint32_t> words,
                                       llvm::SmallVectorImpl<uint64_t> &flits);

// Pack a 64-element INT8 input row in device lane order.
std::array<uint64_t, 16>
packCIMInt8InputCacheRow(llvm::ArrayRef<int8_t> values);

// Validate the six-flit body layout without decoding its numeric values.
llvm::Error validateCIMOutputCacheResponseBody(llvm::ArrayRef<uint64_t> flits);

// Validate padding and sign-extend the sixteen 21-bit output slots.
llvm::Expected<std::array<int32_t, 16>>
decodeCIMInt21OutputCacheResponse(const std::array<uint64_t, 6> &responseFlits);

} // namespace cim22::runtime

#endif // CIM22_RUNTIME_CIMPACKETCODEC_H
