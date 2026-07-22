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

LogicalResult verifyWords(Operation *op, ArrayRef<int32_t> words) {
  // FIXME(CTQ-007): The 256-word address range is provisional.
  if (words.size() != 256)
    return op->emitOpError("expects words to contain 256 i32 values");
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

#define GET_OP_CLASSES
#include "CIM22/Dialect/CIMFrame/IR/CIMFrameOps.cpp.inc"
