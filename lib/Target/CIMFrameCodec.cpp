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
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/bit.h"

#include <array>
#include <cstdint>

namespace mlir::cim22::target {
namespace {
constexpr unsigned kTypeShift = 60;
constexpr std::array<unsigned, 6> kRouteShifts{54, 48, 42, 36, 30, 24};
constexpr uint64_t kControlType = uint64_t{0x8} << kTypeShift;
constexpr uint64_t kWorkOnceType = uint64_t{0x9} << kTypeShift;
constexpr uint64_t kInputType = uint64_t{0x1} << kTypeShift;
constexpr uint64_t kReturnRouteType = uint64_t{0xa} << kTypeShift;
constexpr uint64_t kCacheReadType = uint64_t{0x5} << kTypeShift;
constexpr uint64_t kCIMWriteType = uint64_t{0x2} << kTypeShift;
constexpr uint64_t kWeightExponentType = uint64_t{0x3} << kTypeShift;
constexpr uint64_t kOnce = 1;
constexpr uint64_t kCIMWordCount = 256;
constexpr uint64_t kWeightExponentWordCount = 2;
constexpr uint64_t kConfigBodyCount = 16;
constexpr uint64_t kRequestBit = uint64_t{1} << 23;

uint64_t encodeSignedMagnitude6(int32_t value) {
  const uint64_t magnitude = static_cast<uint64_t>(value < 0 ? -value : value);
  return value < 0 ? magnitude | uint64_t{0x20} : magnitude;
}

uint64_t encodeRoute(ArrayRef<int32_t> route) {
  uint64_t encoded = 0;
  for (auto [value, shift] : llvm::zip_equal(route, kRouteShifts))
    encoded |= encodeSignedMagnitude6(value) << shift;
  return encoded;
}

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
  return (static_cast<uint64_t>(word) << 8) | address;
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
  for (Operation &op : module.getBody()->getOperations()) {
    // Typed packets share their wire format except for the control mode bit.
    llvm::TypeSwitch<Operation *>(&op)
        .Case<cimframe::ControlInt8PacketOp, cimframe::ControlBF16PacketOp>(
            [&](auto control) {
              const uint64_t mode =
                  isa<cimframe::ControlBF16PacketOp>(op) ? 2 : 0;
              flits.push_back(kControlType | encodeRoute(control.getRoute()) |
                              mode | static_cast<uint64_t>(control.getMacro()));
            })
        .Case<cimframe::WorkOncePacketOp>([&](auto work) {
          flits.push_back(kWorkOnceType | encodeRoute(work.getRoute()) | kOnce);
        })
        .Case<cimframe::WriteInputCacheInt8PacketOp,
              cimframe::WriteInputCacheBF16PacketOp>([&](auto input) {
          const uint64_t cacheAddress =
              uint64_t{0x8} | static_cast<uint64_t>(input.getCacheRow());
          flits.push_back(kInputType | encodeRoute(input.getRoute()) |
                          (cacheAddress << 14) | kConfigBodyCount);
          for (int64_t word : input.getWords())
            flits.push_back(static_cast<uint64_t>(word));
        })
        .Case<cimframe::ConfigureTestReturnRoutePacketOp>(
            [&](auto returnRoute) {
              auto testCore = returnRoute.getTestCore();
              flits.push_back(kReturnRouteType |
                              encodeRoute(returnRoute.getRoute()) |
                              (encodeSignedMagnitude6(testCore[0]) << 12) |
                              (encodeSignedMagnitude6(testCore[1]) << 6) |
                              encodeSignedMagnitude6(testCore[2]));
            })
        .Case<cimframe::ReadOutputCacheInt8PacketOp,
              cimframe::ReadOutputCacheBF16PacketOp>([&](auto read) {
          flits.push_back(
              kCacheReadType | encodeRoute(read.getRoute()) | kRequestBit |
              (static_cast<uint64_t>(read.getCacheAddress()) << 14));
        })
        .Case<cimframe::WeightExponentPacketOp>([&](auto exponent) {
          flits.push_back(kWeightExponentType |
                          encodeRoute(exponent.getRoute()) |
                          kWeightExponentWordCount);
          for (int64_t word : exponent.getExponentWords())
            flits.push_back(static_cast<uint64_t>(word));
        })
        .Case<cimframe::CIMInt8WeightPacketOp, cimframe::CIMBF16WeightPacketOp>(
            [&](auto weight) {
              flits.push_back(kCIMWriteType | encodeRoute(weight.getRoute()) |
                              kCIMWordCount);
              for (auto [address, word] : llvm::enumerate(
                       weight.getWords().template getValues<int32_t>())) {
                flits.push_back(
                    encodeCIMWeightPayload(llvm::bit_cast<uint32_t>(word),
                                           static_cast<uint8_t>(address)));
              }
            });
  }
  return flits;
}

FailureOr<SmallVector<uint64_t>> encodeCIMFrameInt8Packets(ModuleOp module) {
  return encodeCIMFramePackets(module);
}

} // namespace mlir::cim22::target
