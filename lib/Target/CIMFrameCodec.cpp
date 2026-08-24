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

#include "llvm/ADT/bit.h"

#include <array>
#include <cstdint>

namespace mlir::cim22::target {
namespace {
constexpr unsigned kTypeShift = 60;
constexpr std::array<unsigned, 6> kRouteShifts{54, 48, 42, 36, 30, 24};
constexpr uint64_t kControlInt8Type = uint64_t{0x8} << kTypeShift;
constexpr uint64_t kWorkOnceType = uint64_t{0x9} << kTypeShift;
constexpr uint64_t kInputInt8Type = uint64_t{0x1} << kTypeShift;
constexpr uint64_t kReturnRouteType = uint64_t{0xa} << kTypeShift;
constexpr uint64_t kCacheReadInt8Type = uint64_t{0x5} << kTypeShift;
constexpr uint64_t kCIMWriteType = uint64_t{0x2} << kTypeShift;
constexpr uint64_t kOnce = 1;
constexpr uint64_t kCIMWordCount = 256;
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
  return isa<cimframe::ControlInt8PacketOp, cimframe::WorkOncePacketOp,
             cimframe::CIMInt8WeightPacketOp,
             cimframe::WriteInputCacheInt8PacketOp,
             cimframe::ConfigureTestReturnRoutePacketOp,
             cimframe::ReadOutputCacheInt8PacketOp>(op);
}

bool isOnecastRoute(ArrayRef<int32_t> route) {
  return route[3] == 0 && route[4] == 0 && route[5] == 0;
}
} // namespace

FailureOr<SmallVector<uint64_t>> encodeCIMFrameInt8Packets(ModuleOp module) {
  if (failed(verify(module)) || failed(cimframe::verifyCIMFrameModule(module)))
    return failure();

  bool hasPacket = false;
  for (Operation &op : module.getBody()->getOperations()) {
    if (op.getName().getDialectNamespace() !=
        cimframe::CIMFrameDialect::getDialectNamespace())
      continue;
    if (!isPacket(op)) {
      op.emitOpError("static INT8 flit encoding requires packet-stage input");
      return failure();
    }
    auto route = op.getAttrOfType<DenseI32ArrayAttr>("route");
    if (!isOnecastRoute(route.asArrayRef())) {
      op.emitOpError(
          "static INT8 flit encoding requires zero Copy route fields");
      return failure();
    }
    hasPacket = true;
  }
  if (!hasPacket) {
    module.emitError(
        "static INT8 flit encoding requires a non-empty packet stage");
    return failure();
  }

  SmallVector<uint64_t> flits;
  for (Operation &op : module.getBody()->getOperations()) {
    if (auto control = dyn_cast<cimframe::ControlInt8PacketOp>(op)) {
      flits.push_back(kControlInt8Type | encodeRoute(control.getRoute()) |
                      static_cast<uint64_t>(control.getMacro()));
      continue;
    }
    if (auto work = dyn_cast<cimframe::WorkOncePacketOp>(op)) {
      flits.push_back(kWorkOnceType | encodeRoute(work.getRoute()) | kOnce);
      continue;
    }
    if (auto input = dyn_cast<cimframe::WriteInputCacheInt8PacketOp>(op)) {
      const uint64_t cacheAddress =
          uint64_t{0x8} | static_cast<uint64_t>(input.getCacheRow());
      flits.push_back(kInputInt8Type | encodeRoute(input.getRoute()) |
                      (cacheAddress << 14) | kConfigBodyCount);
      for (int64_t word : input.getWords())
        flits.push_back(static_cast<uint64_t>(word));
      continue;
    }
    if (auto returnRoute =
            dyn_cast<cimframe::ConfigureTestReturnRoutePacketOp>(op)) {
      auto testCore = returnRoute.getTestCore();
      flits.push_back(
          kReturnRouteType | encodeRoute(returnRoute.getRoute()) |
          (static_cast<uint64_t>(encodeSignedMagnitude6(testCore[0])) << 12) |
          (static_cast<uint64_t>(encodeSignedMagnitude6(testCore[1])) << 6) |
          encodeSignedMagnitude6(testCore[2]));
      continue;
    }
    if (auto read = dyn_cast<cimframe::ReadOutputCacheInt8PacketOp>(op)) {
      flits.push_back(kCacheReadInt8Type | encodeRoute(read.getRoute()) |
                      kRequestBit |
                      (static_cast<uint64_t>(read.getCacheAddress()) << 14));
      continue;
    }
    if (auto weight = dyn_cast<cimframe::CIMInt8WeightPacketOp>(op)) {
      flits.push_back(kCIMWriteType | encodeRoute(weight.getRoute()) |
                      kCIMWordCount);
      for (auto [address, word] :
           llvm::enumerate(weight.getWords().getValues<int32_t>())) {
        const uint64_t data =
            static_cast<uint64_t>(llvm::bit_cast<uint32_t>(word)) << 8;
        flits.push_back(data | static_cast<uint64_t>(address));
      }
    }
  }
  return flits;
}

} // namespace mlir::cim22::target
