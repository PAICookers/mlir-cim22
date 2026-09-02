//===- ExecutionPlanVerifier.cpp - CIM execution-plan checks -*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIM22/Conversion/LinalgToCIM/ExecutionPlanVerifier.h"

#include "CIM22/Dialect/CIM/IR/CIMDialect.h"
#include "CIM22/Dialect/CIM/IR/CIMOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir::cim {
namespace {
const llvm::StringRef kWorkIdentityAttrs[] = {
    "m_tile", "n_tile", "k_tile", "work_id", "group_id", "core_idx",
    "macro_idx", "cim.mapping"};
const llvm::StringRef kGroupIdentityAttrs[] = {"group_id", "core_idx",
                                                "cim.mapping"};

int64_t getI64(Operation *op, StringRef name) {
  return cast<IntegerAttr>(op->getAttr(name)).getInt();
}

LogicalResult requireSame(Operation *expected, Operation *actual,
                          ArrayRef<StringRef> attributes) {
  for (StringRef name : attributes)
    if (expected->getAttr(name) != actual->getAttr(name))
      return actual->emitError("CIM transaction expects '")
             << name << "' to match its work/group identity";
  return success();
}

LogicalResult validateHostOperation(Operation *op) {
  if (op->getName().getDialectNamespace() == "cim")
    return op->emitError(
        "CIM execution-plan operations must be inside cim.transaction");
  if (!isMemoryEffectFree(op))
    return op->emitError("CIM execution plan requires memory-effect-free Host "
                         "SSA between transactions");
  if (op->getNumSuccessors() != 0)
    return op->emitError("CIM execution plan does not support Host control "
                         "flow");
  if (op->getNumRegions() != 0 &&
      op->getName().getDialectNamespace() != "linalg" &&
      op->getName().getStringRef() != "tensor.pad")
    return op->emitError("CIM execution plan does not support this Host "
                         "region op");
  return success();
}

LogicalResult rejectExpected(TransactionOp transaction,
                             ArrayRef<Operation *> operations, size_t cursor,
                             StringRef expected) {
  if (cursor < operations.size())
    return operations[cursor]->emitError("CIM transaction expected ")
           << expected;
  return transaction.emitError("CIM transaction ended before ") << expected;
}

LogicalResult verifyTransaction(TransactionOp transaction) {
  Block &block = transaction.getBody().front();
  SmallVector<Operation *> operations;
  for (Operation &op : block.without_terminator())
    operations.push_back(&op);
  if (operations.empty())
    return transaction.emitError("must contain at least one group");

  struct GroupInfo {
    SmallVector<ConfigureInputOp> inputs;
  };
  SmallVector<GroupInfo> groups;
  SmallVector<int64_t> coreIndices;
  size_t cursor = 0;
  int64_t expectedGroup = 0;
  int64_t expectedWork = 0;

  // Configuration phase: all active Macro inputs and weights are prepared
  // before any core is triggered.
  while (cursor < operations.size() &&
         isa<ConfigureInputOp>(operations[cursor])) {
    GroupInfo group;
    Operation *first = operations[cursor];
    const int64_t groupId = getI64(first, "group_id");
    if (groupId != expectedGroup)
      return first->emitError("CIM transaction expects group_id = ")
             << expectedGroup << ", but got " << groupId;
    while (cursor < operations.size() &&
           isa<ConfigureInputOp>(operations[cursor]) &&
           getI64(operations[cursor], "group_id") == groupId) {
      auto input = cast<ConfigureInputOp>(operations[cursor++]);
      if (cursor >= operations.size() ||
          !isa<ConfigureWeightOp>(operations[cursor]))
        return rejectExpected(transaction, operations, cursor,
                              "cim.configure_weight");
      Operation *weight = operations[cursor++];
      if (failed(requireSame(input, weight, kWorkIdentityAttrs)))
        return failure();
      const int64_t macro = getI64(input, "macro_idx");
      if (getI64(input, "work_id") != expectedWork ||
          macro != static_cast<int64_t>(group.inputs.size()))
        return input.emitError("CIM transaction has non-dense work/Macro "
                               "identity");
      if (auto argument = dyn_cast<BlockArgument>(input.getInput());
          !argument || argument.getOwner() != &block)
        return input.emitError("CIM transaction input must be a transaction "
                               "block argument");
      if (!group.inputs.empty() &&
          failed(requireSame(group.inputs.front(), input,
                             kGroupIdentityAttrs)))
        return failure();
      group.inputs.push_back(input);
      ++expectedWork;
    }
    if (group.inputs.empty() || group.inputs.size() > 2)
      return first->emitError(
          "CIM transaction allows one or two Macros per group");
    const int64_t coreIdx = getI64(first, "core_idx");
    if (llvm::is_contained(coreIndices, coreIdx))
      return first->emitError(
          "CIM transaction allows at most one group per core");
    if (coreIndices.size() == 20)
      return first->emitError(
          "CIM transaction supports at most 20 active cores");
    coreIndices.push_back(coreIdx);
    groups.push_back(std::move(group));
    ++expectedGroup;
  }
  if (groups.empty())
    return rejectExpected(transaction, operations, cursor,
                          "cim.configure_input");

  // Work phase: each configured core dispatches its Macros and is triggered
  // once after every core has completed configuration.
  for (GroupInfo &group : groups) {
    for (ConfigureInputOp input : group.inputs) {
      if (cursor >= operations.size() || !isa<DispatchOp>(operations[cursor]))
        return rejectExpected(transaction, operations, cursor, "cim.dispatch");
      if (failed(requireSame(input, operations[cursor++], kWorkIdentityAttrs)))
        return failure();
    }
    if (cursor >= operations.size() || !isa<OnceOp>(operations[cursor]))
      return rejectExpected(transaction, operations, cursor, "cim.once");
    if (failed(requireSame(group.inputs.front(), operations[cursor++],
                           kGroupIdentityAttrs)))
      return failure();
  }

  // Readback phase: results become Host-visible only through cim.yield.
  SmallVector<Value> expectedResults;
  for (GroupInfo &group : groups) {
    for (ConfigureInputOp input : group.inputs) {
      if (cursor >= operations.size() || !isa<ReadbackOp>(operations[cursor]))
        return rejectExpected(transaction, operations, cursor, "cim.readback");
      auto readback = cast<ReadbackOp>(operations[cursor++]);
      if (failed(requireSame(input, readback, kWorkIdentityAttrs)))
        return failure();
      expectedResults.push_back(readback.getResult());
    }
    if (cursor >= operations.size() ||
        !isa<GroupBarrierOp>(operations[cursor]))
      return rejectExpected(transaction, operations, cursor,
                            "cim.group_barrier");
    if (failed(requireSame(group.inputs.front(), operations[cursor++],
                           ArrayRef<StringRef>{"group_id"})))
      return failure();
  }
  if (cursor != operations.size())
    return operations[cursor]->emitError(
        "CIM transaction has trailing operations after readback phase");

  auto yield = cast<YieldOp>(block.getTerminator());
  if (!llvm::equal(expectedResults, yield.getValues()))
    return yield.emitError("must yield readbacks in transaction work order");
  return success();
}
} // namespace

LogicalResult verifyCIMExecutionPlan(func::FuncOp function) {
  bool hasTransactions = false;
  function.walk([&](TransactionOp) { hasTransactions = true; });
  if (!hasTransactions) {
    if (function->hasAttr("cim.execution_plan_schema_version"))
      return function.emitError(
          "CIM execution plan rejects a schema without cim.transaction");
    return success();
  }
  if (function.isExternal() || !function.getBody().hasOneBlock())
    return function.emitError(
        "CIM execution plan requires one defined straight-line function block");
  auto schema =
      function->getAttrOfType<IntegerAttr>("cim.execution_plan_schema_version");
  if (!schema || !schema.getType().isSignlessInteger(64) ||
      schema.getInt() != 1)
    return function.emitError("CIM execution plan requires "
                              "cim.execution_plan_schema_version = 1 : i64");

  int64_t expectedTransaction = 0;
  for (Operation &op : function.getBody().front()) {
    if (auto transaction = dyn_cast<TransactionOp>(op)) {
      auto index = transaction->getAttrOfType<IntegerAttr>(
          CIMDialect::getTransactionIdxAttrName());
      if (!index || index.getInt() != expectedTransaction)
        return transaction.emitError("CIM execution plan expects "
                                     "cim.transaction_idx = ")
               << expectedTransaction;
      if (failed(verifyTransaction(transaction)))
        return failure();
      ++expectedTransaction;
      continue;
    }
    if (failed(validateHostOperation(&op)))
      return failure();
  }
  return success();
}

} // namespace mlir::cim
