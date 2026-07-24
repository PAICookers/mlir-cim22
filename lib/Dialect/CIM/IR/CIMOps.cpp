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
constexpr llvm::StringLiteral kInvocationAttrs[] = {
    "m_tile",   "n_tile",    "k_tile",     "work_id",
    "group_id", "core_slot", "macro_slot", "cim.mapping"};

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

LogicalResult verifyMapping(Operation *op) {
  auto mapping = op->getAttrOfType<DictionaryAttr>("cim.mapping");
  if (!mapping)
    return op->emitOpError("expects 'cim.mapping' dictionary attribute");
  auto route = mapping.getAs<DenseI64ArrayAttr>("route");
  if (!route || route.size() != 6)
    return op->emitOpError("expects cim.mapping.route with six i64 values");
  for (int64_t value : route.asArrayRef())
    if (value < 0 || value > 31)
      return op->emitOpError("expects route components in [0, 31]");
  if (route[3] != 0 || route[4] != 0 || route[5] != 0)
    return op->emitOpError("expects onecast route with zero Copy fields");
  return success();
}

LogicalResult verifyFullInvocationProvenance(Operation *op) {
  if (countPresent(op, kInvocationAttrs) != std::size(kInvocationAttrs))
    return op->emitOpError("requires complete invocation provenance");
  if (failed(verifyNonNegative(
          op, ArrayRef<llvm::StringLiteral>(kInvocationAttrs).drop_back())))
    return failure();
  auto macro = op->getAttrOfType<IntegerAttr>("macro_slot");
  if (!macro || macro.getInt() < 0 || macro.getInt() > 1)
    return op->emitOpError("expects macro_slot to be 0 or 1");
  return verifyMapping(op);
}

LogicalResult verifyGroupOperation(Operation *op, bool withMapping) {
  auto group = op->getAttrOfType<IntegerAttr>("group_id");
  if (!group || !group.getType().isSignlessInteger(64) || group.getInt() < 0)
    return op->emitOpError("expects non-negative i64 group_id");
  if (!withMapping)
    return countPresent(op, {"m_tile", "n_tile", "k_tile", "work_id",
                             "core_slot", "macro_slot", "cim.mapping"}) == 0
               ? success()
               : op->emitOpError("must carry only group_id provenance");
  auto core = op->getAttrOfType<IntegerAttr>("core_slot");
  if (!core || !core.getType().isSignlessInteger(64) || core.getInt() < 0)
    return op->emitOpError("expects non-negative i64 core_slot");
  if (countPresent(
          op, {"m_tile", "n_tile", "k_tile", "work_id", "macro_slot"}) != 0)
    return op->emitOpError(
        "must carry only group_id, core_slot, and cim.mapping provenance");
  return verifyMapping(op);
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

LogicalResult StaticWeightOp::verify() {
  RankedTensorType type = dyn_cast<RankedTensorType>(getValue().getType());
  if (!type || type.getShape() != llvm::ArrayRef<int64_t>{16, 64} ||
      !type.getElementType().isSignlessInteger(8))
    return emitOpError("expects value type tensor<16x64xi8>");
  if (countPresent(getOperation(), kInvocationAttrs) != 0)
    return emitOpError("must not carry work invocation provenance");
  return success();
}

LogicalResult ConfigureInputOp::verify() {
  RankedTensorType type = getInput().getType();
  if (type.getShape() != llvm::ArrayRef<int64_t>{64} ||
      !type.getElementType().isSignlessInteger(8))
    return emitOpError("expects input type tensor<64xi8>");
  return verifyFullInvocationProvenance(getOperation());
}

LogicalResult ConfigureWeightOp::verify() {
  if (failed(verifyFullInvocationProvenance(getOperation())))
    return failure();
  auto weight = SymbolTable::lookupNearestSymbolFrom<StaticWeightOp>(
      getOperation(), getResourceAttr());
  if (!weight)
    return emitOpError("expects resource to reference cim.static_weight");
  return success();
}

LogicalResult DispatchOp::verify() {
  return verifyFullInvocationProvenance(getOperation());
}

LogicalResult OnceOp::verify() {
  return verifyGroupOperation(getOperation(), /*withMapping=*/true);
}

LogicalResult ReadbackOp::verify() {
  RankedTensorType type = getResult().getType();
  if (type.getShape() != llvm::ArrayRef<int64_t>{16} ||
      !type.getElementType().isSignlessInteger(21))
    return emitOpError("expects result type tensor<16xi21>");
  return verifyFullInvocationProvenance(getOperation());
}

LogicalResult GroupBarrierOp::verify() {
  return verifyGroupOperation(getOperation(), /*withMapping=*/false);
}

#define GET_OP_CLASSES
#include "CIM22/Dialect/CIM/IR/CIMOps.cpp.inc"
