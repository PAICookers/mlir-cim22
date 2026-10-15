//===- CIMFrameCodec.cpp - CIMFrame raw-flit encoding ----------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIM22/Target/CIMFrameCodec.h"

#include "CIM22/Dialect/CIMFrame/IR/CIMFrameDialect.h"
#include "CIM22/Dialect/CIMFrame/IR/CIMFrameOps.h"
#include "CIM22/Runtime/CIMPacketCodec.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/Error.h"

#include <cstdint>

namespace mlir::cim22::target {
namespace {
bool isPacket(Operation &op) {
  return isa<cimframe::ControlInt8PacketOp, cimframe::ControlBF16PacketOp,
             cimframe::WorkOncePacketOp, cimframe::CIMInt8WeightPacketOp,
             cimframe::WeightExponentPacketOp, cimframe::CIMBF16WeightPacketOp,
             cimframe::WriteInputCacheInt8PacketOp,
             cimframe::WriteInputCacheBF16PacketOp,
             cimframe::ConfigureTestReturnRoutePacketOp,
             cimframe::ReadOutputCacheInt8PacketOp,
             cimframe::ReadOutputCacheBF16PacketOp>(op);
}

bool isOnecastRoute(ArrayRef<int32_t> route) {
  return route[3] == 0 && route[4] == 0 && route[5] == 0;
}
} // namespace

uint64_t encodeCIMWeightPayload(uint32_t word, uint8_t address) {
  return ::cim22::runtime::encodeCIMWeightPayload(word, address);
}

FailureOr<SmallVector<uint64_t>> encodeCIMFramePackets(ModuleOp module) {
  if (failed(verify(module)) || failed(cimframe::verifyCIMFrameModule(module)))
    return failure();

  bool hasPacket = false;
  for (Operation &op : module.getBody()->getOperations()) {
    if (op.getName().getDialectNamespace() !=
        cimframe::CIMFrameDialect::getDialectNamespace())
      continue;
    if (!isPacket(op)) {
      op.emitOpError("static flit encoding requires packet-stage input");
      return failure();
    }
    auto route = op.getAttrOfType<DenseI32ArrayAttr>("route");
    if (!isOnecastRoute(route.asArrayRef())) {
      op.emitOpError("static flit encoding requires zero Copy route fields");
      return failure();
    }
    hasPacket = true;
  }
  if (!hasPacket) {
    module.emitError("static flit encoding requires a non-empty packet stage");
    return failure();
  }

  SmallVector<uint64_t> flits;
  llvm::Error encodingError = llvm::Error::success();
  for (Operation &op : module.getBody()->getOperations()) {
    llvm::TypeSwitch<Operation *>(&op)
        .Case<cimframe::ControlInt8PacketOp, cimframe::ControlBF16PacketOp>(
            [&](auto control) {
              const auto dataType = isa<cimframe::ControlBF16PacketOp>(op)
                                        ? ::cim22::execution::CIMDataType::BF16
                                        : ::cim22::execution::CIMDataType::Int8;
              flits.push_back(::cim22::runtime::encodeCIMControlFlit(
                  control.getRoute(), control.getMacro(), dataType));
            })
        .Case<cimframe::WorkOncePacketOp>([&](auto work) {
          flits.push_back(
              ::cim22::runtime::encodeCIMWorkOnceFlit(work.getRoute()));
        })
        .Case<cimframe::WriteInputCacheInt8PacketOp,
              cimframe::WriteInputCacheBF16PacketOp>([&](auto input) {
          if (encodingError)
            return;
          SmallVector<uint64_t, 16> words;
          for (int64_t word : input.getWords())
            words.push_back(static_cast<uint64_t>(word));
          encodingError = ::cim22::runtime::appendCIMInputCachePacket(
              input.getRoute(), input.getCacheRow(), words, flits);
        })
        .Case<cimframe::ConfigureTestReturnRoutePacketOp>(
            [&](auto returnRoute) {
              flits.push_back(::cim22::runtime::encodeCIMReturnRouteFlit(
                  returnRoute.getRoute(), returnRoute.getCoreIdx(),
                  returnRoute.getTestCore()));
            })
        .Case<cimframe::ReadOutputCacheInt8PacketOp,
              cimframe::ReadOutputCacheBF16PacketOp>([&](auto read) {
          flits.push_back(::cim22::runtime::encodeCIMOutputCacheReadFlit(
              read.getRoute(), read.getCacheAddress()));
        })
        .Case<cimframe::WeightExponentPacketOp>([&](auto exponent) {
          if (encodingError)
            return;
          SmallVector<uint64_t, 2> words;
          for (int64_t word : exponent.getExponentWords())
            words.push_back(static_cast<uint64_t>(word));
          encodingError = ::cim22::runtime::appendCIMWeightExponentPacket(
              exponent.getRoute(), words, flits);
        })
        .Case<cimframe::CIMInt8WeightPacketOp, cimframe::CIMBF16WeightPacketOp>(
            [&](auto weight) {
              if (encodingError)
                return;
              SmallVector<uint32_t, 256> words;
              for (int32_t word :
                   weight.getWords().template getValues<int32_t>())
                words.push_back(llvm::bit_cast<uint32_t>(word));
              encodingError = ::cim22::runtime::appendCIMWeightWordsPacket(
                  weight.getRoute(), words, flits);
            });
  }
  if (encodingError) {
    module.emitError(llvm::toString(std::move(encodingError)));
    return failure();
  }
  return flits;
}

FailureOr<SmallVector<uint64_t>> encodeCIMFrameInt8Packets(ModuleOp module) {
  return encodeCIMFramePackets(module);
}

} // namespace mlir::cim22::target
