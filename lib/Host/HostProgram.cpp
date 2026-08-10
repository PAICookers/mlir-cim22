//===- HostProgram.cpp - Host/CIM program projection ----------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIM22/Host/HostProgram.h"

#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::cim22::host {
namespace {
constexpr llvm::StringLiteral kWorkAttrs[] = {
    "m_tile",   "n_tile",    "k_tile",     "work_id",
    "group_id", "core_slot", "macro_slot", "cim.mapping"};

bool isExecutionPlanOp(Operation *op) {
  return isa<cim::ConfigureInputOp, cim::ConfigureWeightOp, cim::DispatchOp,
             cim::OnceOp, cim::ReadbackOp, cim::GroupBarrierOp>(op);
}

FailureOr<int64_t> readI64(Operation *op, StringRef name) {
  auto value = op->getAttrOfType<IntegerAttr>(name);
  if (!value || !value.getType().isSignlessInteger(64) || value.getInt() < 0) {
    op->emitError("HostProgram expects non-negative i64 '") << name << "'";
    return failure();
  }
  return value.getInt();
}

LogicalResult requireSameWork(Operation *expected, Operation *actual) {
  for (StringRef name : kWorkAttrs)
    if (expected->getAttr(name) != actual->getAttr(name))
      return actual->emitError("HostProgram expects '")
             << name << "' to match work provenance";
  return success();
}

LogicalResult validateHostOperation(Operation *op) {
  if (op->getName().getDialectNamespace() == "cim")
    return op->emitError("HostProgram does not recognize this CIM operation");
  if (!isMemoryEffectFree(op))
    return op->emitError("HostProgram requires memory-effect-free Host SSA");
  if (op->getNumSuccessors() != 0)
    return op->emitError("HostProgram does not support Host control flow");
  if (op->getNumRegions() != 0 &&
      op->getName().getDialectNamespace() != "linalg" &&
      op->getName().getStringRef() != "tensor.pad")
    return op->emitError("HostProgram does not support this Host region op");
  return success();
}

LogicalResult validateInputOwner(cim::ConfigureInputOp input) {
  Value owner = input.getInput();
  Block *block = input->getBlock();
  if (auto argument = dyn_cast<BlockArgument>(owner))
    return argument.getOwner() == block
               ? success()
               : input.emitError("HostProgram input owner is foreign");
  Operation *definition = owner.getDefiningOp();
  if (!definition || definition->getBlock() != block ||
      !definition->isBeforeInBlock(input))
    return input.emitError(
        "HostProgram input must be defined before its CIM segment");
  return success();
}

LogicalResult validateReadbackUsers(cim::ReadbackOp readback,
                                    cim::GroupBarrierOp barrier) {
  for (Operation *user : readback->getUsers())
    if (user->getBlock() != barrier->getBlock() ||
        !barrier->isBeforeInBlock(user))
      return user->emitError(
          "HostProgram cannot consume readback before its group barrier");
  return success();
}
} // namespace

FailureOr<HostProgram> buildHostProgram(func::FuncOp function) {
  if (function.isExternal() || !function.getBody().hasOneBlock()) {
    function.emitError("HostProgram requires one defined straight-line block");
    return failure();
  }
  auto schema =
      function->getAttrOfType<IntegerAttr>("cim.execution_plan_schema_version");
  if (!schema || !schema.getType().isSignlessInteger(64) ||
      schema.getInt() != 1) {
    function.emitError(
        "HostProgram requires cim.execution_plan_schema_version = 1 : i64");
    return failure();
  }

  HostProgram program{
      FlatSymbolRefAttr::get(function.getContext(), function.getSymName()),
      function.getFunctionType(),
      {}};
  Block &block = function.getBody().front();
  SmallVector<Operation *> operations;
  for (Operation &op : block)
    operations.push_back(&op);

  SmallVector<Operation *> hostOperations;
  auto flushHostStep = [&] {
    if (hostOperations.empty())
      return;
    program.steps.emplace_back(HostStep{std::move(hostOperations)});
    hostOperations.clear();
  };
  auto rejectExpected = [&](size_t cursor, StringRef expected) {
    if (cursor < operations.size())
      operations[cursor]->emitError("HostProgram found an interleaved Host or "
                                    "out-of-order CIM operation; expected ")
          << expected;
    else
      function.emitError(
          "HostProgram found an incomplete CIM segment; expected ")
          << expected;
    return failure();
  };

  size_t cursor = 0;
  int64_t expectedGroup = 0;
  int64_t expectedWork = 0;
  int64_t nextOrder = 0;
  while (cursor < operations.size()) {
    if (!isExecutionPlanOp(operations[cursor])) {
      if (failed(validateHostOperation(operations[cursor])))
        return failure();
      hostOperations.push_back(operations[cursor++]);
      continue;
    }
    if (!isa<cim::ConfigureInputOp>(operations[cursor]))
      return rejectExpected(cursor, "cim.configure_input");
    flushHostStep();

    size_t segmentBegin = cursor;
    SmallVector<cim::ConfigureInputOp> inputs;
    while (cursor < operations.size() &&
           isa<cim::ConfigureInputOp>(operations[cursor]) &&
           inputs.size() < 2) {
      auto input = cast<cim::ConfigureInputOp>(operations[cursor++]);
      if (cursor >= operations.size() ||
          !isa<cim::ConfigureWeightOp>(operations[cursor]))
        return rejectExpected(cursor, "cim.configure_weight");
      Operation *weight = operations[cursor++];
      if (failed(requireSameWork(input, weight)) ||
          failed(validateInputOwner(input)))
        return failure();
      FailureOr<int64_t> group = readI64(input, "group_id");
      FailureOr<int64_t> work = readI64(input, "work_id");
      FailureOr<int64_t> macro = readI64(input, "macro_slot");
      if (failed(group) || failed(work) || failed(macro))
        return failure();
      if (*group != expectedGroup || *work != expectedWork ||
          *macro != static_cast<int64_t>(inputs.size())) {
        input.emitError("HostProgram requires dense group/work/Macro ordering");
        return failure();
      }
      inputs.push_back(input);
      ++expectedWork;
    }

    for (cim::ConfigureInputOp input : inputs) {
      if (cursor >= operations.size() ||
          !isa<cim::DispatchOp>(operations[cursor]))
        return rejectExpected(cursor, "cim.dispatch");
      if (failed(requireSameWork(input, operations[cursor++])))
        return failure();
    }
    if (cursor >= operations.size() || !isa<cim::OnceOp>(operations[cursor]))
      return rejectExpected(cursor, "cim.once");
    auto once = cast<cim::OnceOp>(operations[cursor++]);
    FailureOr<int64_t> onceGroup = readI64(once, "group_id");
    if (failed(onceGroup) || *onceGroup != expectedGroup) {
      once.emitError("HostProgram once does not match its segment group");
      return failure();
    }

    SmallVector<cim::ReadbackOp> readbacks;
    for (cim::ConfigureInputOp input : inputs) {
      if (cursor >= operations.size() ||
          !isa<cim::ReadbackOp>(operations[cursor]))
        return rejectExpected(cursor, "cim.readback");
      auto readback = cast<cim::ReadbackOp>(operations[cursor++]);
      if (failed(requireSameWork(input, readback)))
        return failure();
      readbacks.push_back(readback);
    }
    if (cursor >= operations.size() ||
        !isa<cim::GroupBarrierOp>(operations[cursor]))
      return rejectExpected(cursor, "cim.group_barrier");
    auto barrier = cast<cim::GroupBarrierOp>(operations[cursor++]);
    FailureOr<int64_t> barrierGroup = readI64(barrier, "group_id");
    if (failed(barrierGroup) || *barrierGroup != expectedGroup) {
      barrier.emitError("HostProgram barrier does not match its segment group");
      return failure();
    }
    for (cim::ReadbackOp readback : readbacks)
      if (failed(validateReadbackUsers(readback, barrier)))
        return failure();

    CIMSegment segment{
        expectedGroup,
        SmallVector<Operation *>(operations.begin() + segmentBegin,
                                 operations.begin() + cursor),
        {},
        {},
        barrier};
    for (auto [slot, input] : llvm::enumerate(inputs)) {
      segment.inputs.push_back(
          {ViewDirection::HostToCIM, input.getInput().getType(),
           input.getInput(), input, barrier, nextOrder++,
           static_cast<int64_t>(slot),
           input->getAttrOfType<IntegerAttr>("work_id").getInt()});
    }
    for (auto [slot, readback] : llvm::enumerate(readbacks)) {
      segment.results.push_back(
          {ViewDirection::CIMToHost, readback.getResult().getType(),
           readback.getResult(), readback, barrier, nextOrder++,
           static_cast<int64_t>(slot),
           readback->getAttrOfType<IntegerAttr>("work_id").getInt()});
    }
    program.steps.emplace_back(std::move(segment));

    if (inputs.size() == 1 &&
        llvm::any_of(ArrayRef<Operation *>(operations).drop_front(cursor),
                     isExecutionPlanOp)) {
      function.emitError(
          "HostProgram permits a single-Macro group only as the final segment");
      return failure();
    }
    ++expectedGroup;
  }
  flushHostStep();
  if (expectedGroup == 0) {
    function.emitError("HostProgram requires at least one CIM segment");
    return failure();
  }
  return program;
}

} // namespace mlir::cim22::host
