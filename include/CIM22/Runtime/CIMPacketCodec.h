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

llvm::Error validateCIMRoute(llvm::ArrayRef<int32_t> route);

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

llvm::Error appendCIMInputCachePacket(llvm::ArrayRef<int32_t> route,
                                      int64_t cacheAddress,
                                      llvm::ArrayRef<uint64_t> words,
                                      llvm::SmallVectorImpl<uint64_t> &flits);
llvm::Error
appendCIMWeightExponentPacket(llvm::ArrayRef<int32_t> route,
                              llvm::ArrayRef<uint64_t> exponentWords,
                              llvm::SmallVectorImpl<uint64_t> &flits);
llvm::Error appendCIMWeightWordsPacket(llvm::ArrayRef<int32_t> route,
                                       llvm::ArrayRef<uint32_t> words,
                                       llvm::SmallVectorImpl<uint64_t> &flits);

std::array<uint64_t, 16>
packCIMInt8InputCacheRow(llvm::ArrayRef<int8_t> values);

llvm::Expected<std::array<int32_t, 16>>
decodeCIMInt21OutputCacheResponse(const std::array<uint64_t, 6> &responseFlits);

} // namespace cim22::runtime

#endif // CIM22_RUNTIME_CIMPACKETCODEC_H
