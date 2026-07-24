//===- CIMOps.cpp - CIM dialect operations ---------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIM22/Dialect/CIM/IR/CIMOps.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::cim;

namespace {
constexpr llvm::StringLiteral kTileAttrs[] = {"m_tile", "n_tile", "k_tile"};
constexpr llvm::StringLiteral kScheduleAttrs[] = {"work_id", "group_id"};
constexpr llvm::StringLiteral kMappingAttrs[] = {"core_slot", "macro_slot",
                                                 "cim.mapping"};

unsigned countPresent(Operation *op, ArrayRef<llvm::StringLiteral> names) {
  return llvm::count_if(names,
                        [op](StringRef name) { return op->hasAttr(name); });
}

LogicalResult verifyNonNegative(Operation *op,
                                ArrayRef<llvm::StringLiteral> names) {
  for (StringRef name : names) {
    Attribute attribute = op->getAttr(name);
    if (!attribute)
      continue;
    auto value = dyn_cast<IntegerAttr>(attribute);
    if (!value || !value.getType().isSignlessInteger(64))
      return op->emitOpError("expects '") << name << "' to be an i64 attribute";
    if (value.getInt() < 0)
      return op->emitOpError("expects '") << name << "' to be non-negative";
  }
  return success();
}
} // namespace

LogicalResult VMMOp::verify() {
  RankedTensorType inputType = getInput().getType();
  RankedTensorType weightType = getWeight().getType();
  RankedTensorType resultType = getResult().getType();

  if (inputType.getShape() != llvm::ArrayRef<int64_t>{64})
    return emitOpError("expects input shape [64], but got ") << inputType;
  if (weightType.getShape() != llvm::ArrayRef<int64_t>{16, 64})
    return emitOpError(
               "expects weight shape [16, 64] in [N, K] order, but got ")
           << weightType;
  if (resultType.getShape() != llvm::ArrayRef<int64_t>{16})
    return emitOpError("expects result shape [16], but got ") << resultType;

  unsigned tileAttrCount = countPresent(getOperation(), kTileAttrs);
  if (tileAttrCount != 0 && tileAttrCount != std::size(kTileAttrs))
    return emitOpError("requires m_tile, n_tile, and k_tile together");
  if (failed(verifyNonNegative(getOperation(), kTileAttrs)))
    return failure();

  unsigned scheduleAttrCount = countPresent(getOperation(), kScheduleAttrs);
  if (scheduleAttrCount != 0 && scheduleAttrCount != std::size(kScheduleAttrs))
    return emitOpError("requires work_id and group_id together");
  if (scheduleAttrCount != 0 && tileAttrCount == 0)
    return emitOpError("requires tile identity before schedule attributes");
  if (failed(verifyNonNegative(getOperation(), kScheduleAttrs)))
    return failure();

  unsigned mappingAttrCount = countPresent(getOperation(), kMappingAttrs);
  if (mappingAttrCount != 0 && mappingAttrCount != std::size(kMappingAttrs))
    return emitOpError(
        "requires core_slot, macro_slot, and cim.mapping together");
  if (mappingAttrCount != 0 && scheduleAttrCount == 0)
    return emitOpError("requires logical schedule before target mapping");
  if (failed(verifyNonNegative(
          getOperation(),
          ArrayRef<llvm::StringLiteral>(kMappingAttrs).take_front(2))))
    return failure();
  if (Attribute mapping = getOperation()->getAttr("cim.mapping");
      mapping && !isa<DictionaryAttr>(mapping))
    return emitOpError("expects 'cim.mapping' to be a dictionary attribute");
  return success();
}

#define GET_OP_CLASSES
#include "CIM22/Dialect/CIM/IR/CIMOps.cpp.inc"
