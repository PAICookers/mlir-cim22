//===- ExecutionPlanVerifier.cpp - CIM execution-plan checks -*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIM22/Conversion/LinalgToCIM/ExecutionPlanVerifier.h"
#include "CIM22/Conversion/LinalgToCIM/CIMSegments.h"

#include "CIM22/Dialect/CIM/IR/CIMDialect.h"
#include "CIM22/Dialect/CIM/IR/CIMOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir::cim {
namespace {
const llvm::StringRef kWorkIdentityAttrs[] = {
    CIMDialect::getSegmentIdAttrName(),
    "m_tile",
    "n_tile",
    "k_tile",
    "work_id",
    "group_id",
    "core_slot",
    "macro_slot",
    "cim.mapping"};
const llvm::StringRef kGroupIdentityAttrs[] = {
    CIMDialect::getSegmentIdAttrName(), "group_id", "core_slot", "cim.mapping"};
const llvm::StringRef kBarrierIdentityAttrs[] = {
    CIMDialect::getSegmentIdAttrName(), "group_id"};

int64_t getI64(Operation *op, StringRef name) {
  return cast<IntegerAttr>(op->getAttr(name)).getInt();
}

LogicalResult requireSameWork(Operation *expected, Operation *actual) {
  for (StringRef name : kWorkIdentityAttrs)
    if (expected->getAttr(name) != actual->getAttr(name))
      return actual->emitError("CIM execution plan expects '")
             << name << "' to match work identity";
  return success();
}

LogicalResult requireSameGroup(Operation *expected, Operation *actual) {
  for (StringRef name : kGroupIdentityAttrs)
    if (expected->getAttr(name) != actual->getAttr(name))
      return actual->emitError("CIM execution plan expects '")
             << name << "' to match group identity";
  return success();
}

LogicalResult requireSameBarrier(Operation *expected, Operation *actual) {
  for (StringRef name : kBarrierIdentityAttrs)
    if (expected->getAttr(name) != actual->getAttr(name))
      return actual->emitError("CIM execution plan expects '")
             << name << "' to match group identity";
  return success();
}

LogicalResult validateHostOperation(Operation *op) {
  if (op->getName().getDialectNamespace() == "cim")
    return op->emitError("CIM execution plan does not recognize this CIM "
                         "operation");
  if (!isMemoryEffectFree(op))
    return op->emitError(
        "CIM execution plan requires memory-effect-free Host SSA");
  if (op->getNumSuccessors() != 0)
    return op->emitError("CIM execution plan does not support Host control "
                         "flow");
  if (op->getNumRegions() != 0 &&
      op->getName().getDialectNamespace() != "linalg" &&
      op->getName().getStringRef() != "tensor.pad")
    return op->emitError(
        "CIM execution plan does not support this Host region op");
  return success();
}

LogicalResult validateInputOwner(ConfigureInputOp input,
                                 Operation *segmentStart) {
  Value owner = input.getInput();
  Block *block = input->getBlock();
  if (auto argument = dyn_cast<BlockArgument>(owner))
    return argument.getOwner() == block
               ? success()
               : input.emitError("CIM execution-plan input owner is foreign");
  Operation *definition = owner.getDefiningOp();
  if (!definition || definition->getBlock() != block ||
      !definition->isBeforeInBlock(segmentStart))
    return input.emitError(
               "CIM execution-plan input must be available before segment ")
           << getI64(segmentStart, CIMDialect::getSegmentIdAttrName())
           << " starts";
  return success();
}

LogicalResult validateReadbackUsers(ReadbackOp readback,
                                    GroupBarrierOp barrier) {
  for (Operation *user : readback->getUsers())
    if (user->getBlock() != barrier->getBlock() ||
        !barrier->isBeforeInBlock(user))
      return user->emitError(
          "CIM execution plan cannot consume readback before its group "
          "barrier");
  return success();
}

LogicalResult rejectExpected(func::FuncOp function,
                             ArrayRef<Operation *> operations, size_t cursor,
                             StringRef expected) {
  if (cursor < operations.size())
    operations[cursor]->emitError(
        "CIM execution plan found an interleaved Host or "
        "out-of-order CIM operation; expected ")
        << expected;
  else
    function.emitError("CIM execution plan found an incomplete group; "
                       "expected ")
        << expected;
  return failure();
}
} // namespace

LogicalResult verifyCIMExecutionPlan(func::FuncOp function) {
  bool hasExecutionPlan = false;
  function.walk(
      [&](Operation *op) { hasExecutionPlan |= isExecutionPlanOp(op); });
  if (!hasExecutionPlan) {
    if (function->hasAttr("cim.execution_plan_schema_version"))
      return function.emitError(
          "CIM execution plan rejects a schema without plan operations");
    return success();
  }

  if (function.isExternal() || !function.getBody().hasOneBlock())
    return function.emitError(
        "CIM execution plan requires one defined straight-line block");
  auto schema =
      function->getAttrOfType<IntegerAttr>("cim.execution_plan_schema_version");
  if (!schema || !schema.getType().isSignlessInteger(64) ||
      schema.getInt() != 1)
    return function.emitError(
        "CIM execution plan requires cim.execution_plan_schema_version = 1 "
        ": i64");

  Block &block = function.getBody().front();
  SmallVector<Operation *> operations;
  for (Operation &op : block)
    operations.push_back(&op);

  size_t cursor = 0;
  int64_t currentSegment = -1;
  int64_t expectedGroup = 0;
  int64_t expectedWork = 0;
  Operation *segmentStart = nullptr;
  bool segmentClosed = false;
  while (cursor < operations.size()) {
    if (!isExecutionPlanOp(operations[cursor])) {
      if (failed(validateHostOperation(operations[cursor])))
        return failure();
      ++cursor;
      continue;
    }
    if (!isa<ConfigureInputOp>(operations[cursor]))
      return rejectExpected(function, operations, cursor,
                            "cim.configure_input");

    int64_t segmentId =
        getI64(operations[cursor], CIMDialect::getSegmentIdAttrName());
    if (segmentId != currentSegment) {
      if (segmentId != currentSegment + 1)
        return operations[cursor]->emitError(
                   "CIM execution plan expects cim.segment_id = ")
               << currentSegment + 1 << ", but got " << segmentId;
      currentSegment = segmentId;
      expectedGroup = 0;
      expectedWork = 0;
      segmentStart = operations[cursor];
      segmentClosed = false;
    } else if (segmentClosed) {
      return operations[cursor]->emitError(
          "CIM execution plan permits a single-Macro group only as the final "
          "group of its segment");
    }

    SmallVector<ConfigureInputOp> inputs;
    while (cursor < operations.size() &&
           isa<ConfigureInputOp>(operations[cursor]) && inputs.size() < 2) {
      auto input = cast<ConfigureInputOp>(operations[cursor++]);
      if (cursor >= operations.size() ||
          !isa<ConfigureWeightOp>(operations[cursor]))
        return rejectExpected(function, operations, cursor,
                              "cim.configure_weight");
      Operation *weight = operations[cursor++];
      if (failed(requireSameWork(input, weight)) ||
          failed(validateInputOwner(input, segmentStart)))
        return failure();
      int64_t group = getI64(input, "group_id");
      int64_t work = getI64(input, "work_id");
      int64_t macro = getI64(input, "macro_slot");
      if (group != expectedGroup || work != expectedWork ||
          macro != static_cast<int64_t>(inputs.size()))
        return input.emitError(
                   "CIM execution plan expects segment/group/work/Macro = ")
               << currentSegment << "/" << expectedGroup << "/" << expectedWork
               << "/" << inputs.size();
      if (!inputs.empty() && failed(requireSameGroup(inputs.front(), input)))
        return failure();
      inputs.push_back(input);
      ++expectedWork;
    }
    for (ConfigureInputOp input : inputs) {
      if (cursor >= operations.size() || !isa<DispatchOp>(operations[cursor]))
        return rejectExpected(function, operations, cursor, "cim.dispatch");
      if (failed(requireSameWork(input, operations[cursor++])))
        return failure();
    }
    if (cursor >= operations.size() || !isa<OnceOp>(operations[cursor]))
      return rejectExpected(function, operations, cursor, "cim.once");
    auto once = cast<OnceOp>(operations[cursor++]);
    if (failed(requireSameGroup(inputs.front(), once)))
      return failure();

    SmallVector<ReadbackOp> readbacks;
    for (ConfigureInputOp input : inputs) {
      if (cursor >= operations.size() || !isa<ReadbackOp>(operations[cursor]))
        return rejectExpected(function, operations, cursor, "cim.readback");
      auto readback = cast<ReadbackOp>(operations[cursor++]);
      if (failed(requireSameWork(input, readback)))
        return failure();
      readbacks.push_back(readback);
    }
    if (cursor >= operations.size() || !isa<GroupBarrierOp>(operations[cursor]))
      return rejectExpected(function, operations, cursor, "cim.group_barrier");
    auto barrier = cast<GroupBarrierOp>(operations[cursor++]);
    if (failed(requireSameBarrier(inputs.front(), barrier)))
      return failure();
    for (ReadbackOp readback : readbacks)
      if (failed(validateReadbackUsers(readback, barrier)))
        return failure();

    segmentClosed = inputs.size() == 1;
    ++expectedGroup;
  }
  return success();
}

} // namespace mlir::cim
