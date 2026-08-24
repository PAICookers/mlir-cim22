//===- CIMFrameOps.cpp - CIMFrame operations ------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIM22/Dialect/CIMFrame/IR/CIMFrameOps.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::cimframe;

namespace {
LogicalResult verifyProtocolAttributes(Operation *op,
                                       ArrayRef<StringRef> allowed) {
  for (NamedAttribute attribute : op->getAttrs()) {
    StringRef name = attribute.getName().strref();
    if (name.contains('.') || llvm::is_contained(allowed, name))
      continue;
    return op->emitOpError("unexpected protocol attribute '") << name << "'";
  }
  return success();
}

LogicalResult verifyRoute(Operation *op, ArrayRef<int32_t> route) {
  if (route.size() != 6)
    return op->emitOpError("expects route to contain 6 signed distances");
  if (llvm::any_of(route,
                   [](int32_t value) { return value < -31 || value > 31; }))
    return op->emitOpError("expects each route distance in [-31, 31]");
  return success();
}

LogicalResult verifyMacro(Operation *op, uint32_t macro) {
  if (macro > 1)
    return op->emitOpError("expects macro to be 0 or 1");
  return success();
}

LogicalResult verifyInputWords(Operation *op, ArrayRef<int64_t> words) {
  if (words.size() != 16)
    return op->emitOpError("expects input Cache row to contain 16 i64 values");
  return success();
}

LogicalResult verifyWords(Operation *op, DenseIntElementsAttr words) {
  if (!words.getElementType().isSignlessInteger(32) ||
      words.getNumElements() != 256)
    return op->emitOpError("expects words to contain 256 i32 values");
  return success();
}

LogicalResult verifyCacheRow(Operation *op, int64_t row) {
  if (row < 0 || row > 7)
    return op->emitOpError("expects input Cache row to be in [0, 7]");
  return success();
}

LogicalResult verifyCacheAddress(Operation *op, int64_t address) {
  if (address < 0 || address > 7)
    return op->emitOpError("expects output Cache address to be in [0, 7]");
  return success();
}

LogicalResult verifyTestCore(Operation *op, ArrayRef<int32_t> route) {
  if (route.size() != 3)
    return op->emitOpError("expects test_core to contain 3 signed distances");
  if (llvm::any_of(route,
                   [](int32_t value) { return value < -31 || value > 31; }))
    return op->emitOpError("expects each test_core distance in [-31, 31]");
  return success();
}

LogicalResult verifyRoutedOperation(Operation *op, ArrayRef<int32_t> route,
                                    ArrayRef<StringRef> allowed) {
  if (failed(verifyProtocolAttributes(op, allowed)))
    return failure();
  return verifyRoute(op, route);
}
} // namespace

LogicalResult StartInt8OnceOp::verify() {
  if (failed(verifyRoutedOperation(getOperation(), getRoute(),
                                   {"route", "macro"})))
    return failure();
  return verifyMacro(getOperation(), getMacro());
}

LogicalResult WriteInt8WeightsOp::verify() {
  if (failed(verifyRoutedOperation(getOperation(), getRoute(),
                                   {"route", "macro", "words"})))
    return failure();
  if (failed(verifyMacro(getOperation(), getMacro())))
    return failure();
  return verifyWords(getOperation(), getWords());
}

LogicalResult ControlInt8PacketOp::verify() {
  if (failed(verifyRoutedOperation(getOperation(), getRoute(),
                                   {"route", "macro"})))
    return failure();
  return verifyMacro(getOperation(), getMacro());
}

LogicalResult WorkOncePacketOp::verify() {
  return verifyRoutedOperation(getOperation(), getRoute(), {"route"});
}

LogicalResult CIMInt8WeightPacketOp::verify() {
  if (failed(verifyRoutedOperation(getOperation(), getRoute(),
                                   {"route", "words"})))
    return failure();
  return verifyWords(getOperation(), getWords());
}

LogicalResult WriteInputCacheInt8PacketOp::verify() {
  if (failed(verifyRoutedOperation(getOperation(), getRoute(),
                                   {"route", "macro", "cache_row", "words"})))
    return failure();
  if (failed(verifyMacro(getOperation(), getMacro())) ||
      failed(verifyCacheRow(getOperation(), getCacheRow())))
    return failure();
  return verifyInputWords(getOperation(), getWords());
}

LogicalResult ConfigureTestReturnRoutePacketOp::verify() {
  if (failed(verifyRoutedOperation(getOperation(), getRoute(),
                                   {"route", "test_core"})))
    return failure();
  return verifyTestCore(getOperation(), getTestCore());
}

LogicalResult ReadOutputCacheInt8PacketOp::verify() {
  if (failed(verifyRoutedOperation(getOperation(), getRoute(),
                                   {"route", "cache_address"})))
    return failure();
  return verifyCacheAddress(getOperation(), getCacheAddress());
}

LogicalResult mlir::cimframe::verifyCIMFrameModule(ModuleOp module) {
  Block::OpListType &operations = module.getBody()->getOperations();
  Operation *firstCommand = nullptr;
  Operation *firstPacket = nullptr;
  for (Operation &op : operations) {
    if (isa<StartInt8OnceOp, WriteInt8WeightsOp>(op))
      firstCommand = firstCommand ? firstCommand : &op;
    if (isa<ControlInt8PacketOp, WorkOncePacketOp, CIMInt8WeightPacketOp,
            WriteInputCacheInt8PacketOp, ConfigureTestReturnRoutePacketOp,
            ReadOutputCacheInt8PacketOp>(op))
      firstPacket = firstPacket ? firstPacket : &op;
  }

  if (firstCommand && firstPacket)
    return module.emitError(
        "cannot mix cimframe command and packet stages in one module");
  if (!firstPacket)
    return success();

  bool invalid = false;
  for (Operation &op : operations) {
    if (auto control = dyn_cast<ControlInt8PacketOp>(op)) {
      Operation *next = op.getNextNode();
      if (auto work = dyn_cast_or_null<WorkOncePacketOp>(next)) {
        if (control.getRoute() != work.getRoute()) {
          work.emitOpError("expects control/work routes to match");
          invalid = true;
        }
      } else if (auto weight = dyn_cast_or_null<CIMInt8WeightPacketOp>(next)) {
        if (control.getRoute() != weight.getRoute()) {
          weight.emitOpError("expects control/weight routes to match");
          invalid = true;
        }
      } else if (auto input =
                     dyn_cast_or_null<WriteInputCacheInt8PacketOp>(next)) {
        if (control.getRoute() != input.getRoute() ||
            control.getMacro() != input.getMacro()) {
          input.emitOpError("expects control/input route and Macro to match");
          invalid = true;
        }
      } else if (auto read =
                     dyn_cast_or_null<ReadOutputCacheInt8PacketOp>(next)) {
        auto previous = dyn_cast_or_null<ConfigureTestReturnRoutePacketOp>(
            op.getPrevNode());
        if (!previous || previous.getRoute() != control.getRoute() ||
            read.getRoute() != control.getRoute()) {
          control.emitOpError("expects test return route, control and output "
                              "read routes to match");
          invalid = true;
        }
      } else {
        control.emitOpError(
            "expects control_int8_packet immediately followed by "
            "work_once_packet or cim_int8_weight_packet; input and readback "
            "packets must use their matching typed sequence");
        invalid = true;
      }
      continue;
    }
    if (auto work = dyn_cast<WorkOncePacketOp>(op)) {
      if (!isa_and_nonnull<ControlInt8PacketOp>(op.getPrevNode())) {
        work.emitOpError("expects work_once_packet immediately preceded by "
                         "control_int8_packet");
        invalid = true;
      }
      continue;
    }
    if (auto weight = dyn_cast<CIMInt8WeightPacketOp>(op)) {
      if (!isa_and_nonnull<ControlInt8PacketOp>(op.getPrevNode())) {
        weight.emitOpError(
            "expects cim_int8_weight_packet immediately preceded by "
            "control_int8_packet");
        invalid = true;
      }
      continue;
    }
    if (auto input = dyn_cast<WriteInputCacheInt8PacketOp>(op)) {
      if (!isa_and_nonnull<ControlInt8PacketOp>(op.getPrevNode())) {
        input.emitOpError(
            "expects write_input_cache_int8_packet immediately preceded by "
            "control_int8_packet");
        invalid = true;
      }
      continue;
    }
    if (auto route = dyn_cast<ConfigureTestReturnRoutePacketOp>(op)) {
      if (!isa_and_nonnull<ControlInt8PacketOp>(
              route.getOperation()->getNextNode())) {
        route.emitOpError("expects configure_test_return_route_packet "
                          "immediately followed by "
                          "control_int8_packet");
        invalid = true;
      }
      continue;
    }
    if (auto read = dyn_cast<ReadOutputCacheInt8PacketOp>(op)) {
      auto control = dyn_cast_or_null<ControlInt8PacketOp>(op.getPrevNode());
      auto route = control ? dyn_cast_or_null<ConfigureTestReturnRoutePacketOp>(
                                 control->getPrevNode())
                           : nullptr;
      if (!control || !route) {
        read.emitOpError("expects read_output_cache_int8_packet after "
                         "configure_test_return_route_packet "
                         "and control_int8_packet");
        invalid = true;
      }
    }
  }
  return failure(invalid);
}

#define GET_OP_CLASSES
#include "CIM22/Dialect/CIMFrame/IR/CIMFrameOps.cpp.inc"
