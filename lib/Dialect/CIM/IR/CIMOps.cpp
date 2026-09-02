//===- CIMOps.cpp - CIM dialect operations ---------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIM22/Dialect/CIM/IR/CIMOps.h"

#include "CIM22/Dialect/CIM/IR/CIMDialect.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::cim;

namespace {
const llvm::StringRef kTileAttrs[] = {"m_tile", "n_tile", "k_tile"};
const llvm::StringRef kScheduleAttrs[] = {"work_id", "group_id"};
const llvm::StringRef kMappingAttrs[] = {"core_idx", "macro_idx",
                                         "cim.mapping"};
const llvm::StringRef kExecutionPlanIdentityAttrs[] = {
    "m_tile",
    "n_tile",
    "k_tile",
    "work_id",
    "group_id",
    "core_idx",
    "macro_idx",
    "cim.mapping"};

unsigned countPresent(Operation *op, ArrayRef<llvm::StringRef> names) {
  return llvm::count_if(names,
                        [op](StringRef name) { return op->hasAttr(name); });
}

LogicalResult verifyNonNegative(Operation *op,
                                ArrayRef<llvm::StringRef> names) {
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
    if (!llvm::isUInt<5>(value))
      return op->emitOpError("expects route components in [0, 31]");
  if (route[3] != 0 || route[4] != 0 || route[5] != 0)
    return op->emitOpError("expects onecast route with zero Copy fields");
  return success();
}

LogicalResult verifyFullExecutionPlanIdentity(Operation *op) {
  if (countPresent(op, kExecutionPlanIdentityAttrs) !=
      std::size(kExecutionPlanIdentityAttrs))
    return op->emitOpError("requires complete execution-plan identity");
  if (failed(verifyNonNegative(
          op, ArrayRef(kExecutionPlanIdentityAttrs).drop_back())))
    return failure();
  auto macro = op->getAttrOfType<IntegerAttr>("macro_idx");
  if (!macro || macro.getInt() < 0 || macro.getInt() > 1)
    return op->emitOpError("expects macro_idx to be 0 or 1");
  return verifyMapping(op);
}

LogicalResult verifyGroupOperation(Operation *op, bool withMapping) {
  auto group = op->getAttrOfType<IntegerAttr>("group_id");
  if (!group || !group.getType().isSignlessInteger(64) || group.getInt() < 0)
    return op->emitOpError("expects non-negative i64 group_id");
  if (!withMapping)
    return countPresent(op, {"m_tile", "n_tile", "k_tile", "work_id",
                             "core_idx", "macro_idx", "cim.mapping"}) == 0
               ? success()
               : op->emitOpError(
                     "must carry only group_id identity");
  auto core = op->getAttrOfType<IntegerAttr>("core_idx");
  if (!core || !core.getType().isSignlessInteger(64) || core.getInt() < 0)
    return op->emitOpError("expects non-negative i64 core_idx");
  if (countPresent(
          op, {"m_tile", "n_tile", "k_tile", "work_id", "macro_idx"}) != 0)
    return op->emitOpError(
        "must carry only group_id, core_idx, and cim.mapping "
        "identity");
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
        "requires core_idx, macro_idx, and cim.mapping together");
  if (mappingAttrCount != 0 && scheduleAttrCount == 0)
    return emitOpError("requires logical schedule before target mapping");
  if (failed(verifyNonNegative(getOperation(),
                               ArrayRef(kMappingAttrs).take_front(2))))
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
  if (countPresent(getOperation(), kExecutionPlanIdentityAttrs) != 0)
    return emitOpError("must not carry per-work execution-plan identity");
  return success();
}

LogicalResult ConfigureInputOp::verify() {
  RankedTensorType type = getInput().getType();
  if (type.getShape() != llvm::ArrayRef<int64_t>{64} ||
      !type.getElementType().isSignlessInteger(8))
    return emitOpError("expects input type tensor<64xi8>");
  return verifyFullExecutionPlanIdentity(getOperation());
}

LogicalResult ConfigureWeightOp::verify() {
  if (failed(verifyFullExecutionPlanIdentity(getOperation())))
    return failure();
  auto weight = SymbolTable::lookupNearestSymbolFrom<StaticWeightOp>(
      getOperation(), getResourceAttr());
  if (!weight)
    return emitOpError("expects resource to reference cim.static_weight");
  return success();
}

LogicalResult DispatchOp::verify() {
  return verifyFullExecutionPlanIdentity(getOperation());
}

LogicalResult OnceOp::verify() {
  return verifyGroupOperation(getOperation(), /*withMapping=*/true);
}

LogicalResult ReadbackOp::verify() {
  RankedTensorType type = getResult().getType();
  if (type.getShape() != llvm::ArrayRef<int64_t>{16} ||
      !type.getElementType().isSignlessInteger(21))
    return emitOpError("expects result type tensor<16xi21>");
  return verifyFullExecutionPlanIdentity(getOperation());
}

LogicalResult GroupBarrierOp::verify() {
  return verifyGroupOperation(getOperation(), /*withMapping=*/false);
}

LogicalResult TransactionOp::verify() {
  auto index = getOperation()->getAttrOfType<IntegerAttr>(
      CIMDialect::getTransactionIdxAttrName());
  if (!index || !index.getType().isSignlessInteger(64) || index.getInt() < 0)
    return emitOpError("expects non-negative i64 cim.transaction_idx");
  if (!getBody().hasOneBlock())
    return emitOpError("expects exactly one body block");
  Block &block = getBody().front();
  if (block.getNumArguments() != getInputs().size())
    return emitOpError("expects one region argument for each input");
  for (auto [argument, input] : llvm::zip(block.getArguments(), getInputs()))
    if (argument.getType() != input.getType())
      return emitOpError("transaction input and region argument types differ");
  auto yield = dyn_cast<YieldOp>(block.getTerminator());
  if (!yield || yield.getNumOperands() != getNumResults())
    return emitOpError("expects cim.yield operands to match transaction results");
  for (auto [result, value] : llvm::zip(getResults(), yield.getOperands()))
    if (result.getType() != value.getType())
      return emitOpError("transaction result and yield types differ");
  for (Operation &op : block.without_terminator())
    if (!isa<ConfigureInputOp, ConfigureWeightOp, DispatchOp, OnceOp,
             ReadbackOp, GroupBarrierOp>(&op))
      return op.emitError("Host operations are not allowed inside cim.transaction");
  return success();
}

#define GET_OP_CLASSES
#include "CIM22/Dialect/CIM/IR/CIMOps.cpp.inc"
