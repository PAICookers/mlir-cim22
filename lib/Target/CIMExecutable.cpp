//===- CIMTransaction.cpp - CIM22 target executable compiler -----*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIM22/Target/CIMExecutable.h"

#include "CIM22/Conversion/LinalgToCIM/CIMSegments.h"
#include "CIM22/Conversion/LinalgToCIM/ExecutionPlanVerifier.h"
#include "CIM22/Dialect/CIM/IR/CIMDialect.h"
#include "CIM22/Dialect/CIM/IR/CIMOps.h"
#include "CIM22/Dialect/CIMFrame/IR/CIMFrameDialect.h"
#include "CIM22/Dialect/CIMFrame/IR/CIMFrameOps.h"
#include "CIM22/Support/Int8WeightLayout.h"
#include "CIM22/Target/CIMFrameCodec.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/bit.h"

#include <array>
#include <cstdint>

namespace mlir::cim22::target {
using ::cim22::execution::CIMTransaction;
using ::cim22::execution::CIMFramePacket;
using ::cim22::execution::CIMGroup;
using ::cim22::execution::CIMPacketKind;
using ::cim22::execution::CIMWork;
using ::cim22::execution::DynamicInputBinding;
using ::cim22::execution::ReadbackBinding;
using ::cim22::execution::StaticWeightSection;
namespace {
constexpr llvm::StringLiteral kProfileId = "cim22-4x5-v1";
constexpr int64_t kProfileVersion = 1;
constexpr int64_t kExecutionPlanVersion = 1;
constexpr llvm::StringLiteral kOnecastRoutePolicy = "lower-left-maximal-xy-v1";

bool isExecutionPlanFunction(func::FuncOp function) {
  bool found = false;
  function.walk([&](Operation *op) {
    found |= isa<cim::ConfigureInputOp, cim::ConfigureWeightOp, cim::DispatchOp,
                 cim::OnceOp, cim::ReadbackOp, cim::GroupBarrierOp>(op);
  });
  return found;
}

FailureOr<int64_t> readI64(Operation *op, StringRef name) {
  auto value = op->getAttrOfType<IntegerAttr>(name);
  if (!value || !value.getType().isSignlessInteger(64) || value.getInt() < 0) {
    op->emitError("CIM executable requires non-negative i64 '") << name << "'";
    return failure();
  }
  return value.getInt();
}

FailureOr<std::array<int32_t, 6>> readRoute(Operation *op,
                                            StringRef mappingName) {
  auto mapping = op->getAttrOfType<DictionaryAttr>(mappingName);
  auto route = mapping ? mapping.get("route") : Attribute();
  auto values = dyn_cast_or_null<DenseI64ArrayAttr>(route);
  if (!values || values.size() != 6) {
    op->emitError("CIM executable requires a six-element mapping route");
    return failure();
  }
  std::array<int32_t, 6> result{};
  for (auto [index, value] : llvm::enumerate(values.asArrayRef())) {
    if (value < -31 || value > 31 || (index >= 3 && value != 0)) {
      op->emitError("CIM executable requires a zero-Copy onecast route");
      return failure();
    }
    result[index] = static_cast<int32_t>(value);
  }
  return result;
}

FailureOr<std::array<int32_t, 3>> readTestCore(Operation *op) {
  std::array<int32_t, 3> result{};
  constexpr StringLiteral names[] = {"test_core_xy", "test_core_x",
                                     "test_core_y"};
  for (auto [index, name] : llvm::enumerate(names)) {
    auto value = op->getAttrOfType<IntegerAttr>(name);
    if (!value || !value.getType().isSignlessInteger(64) ||
        value.getInt() < -31 || value.getInt() > 31) {
      op->emitError("CIM executable requires '")
          << name << "' as an i64 distance in [-31, 31]";
      return failure();
    }
    result[index] = static_cast<int32_t>(value.getInt());
  }
  return result;
}

FailureOr<int64_t> readOutputAddress(cim::ReadbackOp op) {
  auto value = op->getAttrOfType<IntegerAttr>("output_cache_address");
  if (!value || !value.getType().isSignlessInteger(64) || value.getInt() < 0 ||
      value.getInt() > 7) {
    op.emitError(
        "CIM executable requires output_cache_address in [0, 7] : i64");
    return failure();
  }
  return value.getInt();
}

FailureOr<StaticWeightSection *>
findWeight(std::vector<StaticWeightSection> &weights, int64_t groupId,
           int64_t workId) {
  auto it = llvm::find_if(weights, [&](const StaticWeightSection &weight) {
    return weight.groupId == groupId && weight.workId == workId;
  });
  if (it == weights.end())
    return failure();
  return &*it;
}

FailureOr<CIMGroup *> findGroup(std::vector<CIMGroup> &groups,
                                int64_t groupId) {
  auto it = llvm::find_if(
      groups, [&](const CIMGroup &group) { return group.groupId == groupId; });
  if (it == groups.end())
    return failure();
  return &*it;
}

FailureOr<CIMWork *> findWork(CIMGroup &group, int64_t workId) {
  auto it = llvm::find_if(
      group.works, [&](const CIMWork &work) { return work.workId == workId; });
  if (it == group.works.end())
    return failure();
  return &*it;
}

FailureOr<StaticWeightSection>
buildStaticWeight(cimframe::CIMInt8WeightPacketOp packet) {
  auto planBinding = packet->getAttrOfType<DictionaryAttr>(
      cim::CIMDialect::getPlanBindingAttrName());
  if (!planBinding)
    return packet.emitError(
        "CIM executable requires cim.plan_binding on weight packet");
  auto get = [&](StringRef name) -> FailureOr<int64_t> {
    auto value = planBinding.getAs<IntegerAttr>(name);
    if (!value || !value.getType().isSignlessInteger(64) || value.getInt() < 0)
      return failure();
    return value.getInt();
  };
  auto group = get("group_id");
  auto work = get("work_id");
  auto core = get("core_idx");
  auto macro = get("macro_idx");
  auto resource = planBinding.getAs<FlatSymbolRefAttr>("resource");
  if (failed(group) || failed(work) || failed(core) || failed(macro) ||
      !resource)
    return packet.emitError(
        "CIM executable requires complete weight plan binding");
  std::vector<int32_t> words(packet.getWords().getValues<int32_t>().begin(),
                             packet.getWords().getValues<int32_t>().end());
  std::array<uint32_t, 256> rawWords{};
  for (auto [index, word] :
       llvm::enumerate(packet.getWords().getValues<int32_t>()))
    rawWords[index] = llvm::bit_cast<uint32_t>(word);
  const std::array<uint8_t, 16 * 64> bytes =
      mlir::cim22::unmapCIMWordsToInt8WeightTile(rawWords);

  StaticWeightSection result{
      *group, *work,           *core, *macro, resource.getValue().str(),
      {},     std::move(words)};
  for (auto [index, byte] : llvm::enumerate(bytes))
    result.values[index] = static_cast<int8_t>(byte);
  return result;
}
} // namespace

FailureOr<CIMTransaction> compileCIMTransaction(ModuleOp module) {
  if (failed(verify(module)))
    return failure();

  SmallVector<func::FuncOp> plans;
  for (func::FuncOp function : module.getOps<func::FuncOp>())
    if (isExecutionPlanFunction(function))
      plans.push_back(function);
  if (plans.size() != 1) {
    module.emitError("compile-cim-executable expects exactly one "
                     "execution-plan function");
    return failure();
  }
  func::FuncOp function = plans.front();
  if (failed(cim::verifyCIMExecutionPlan(function)))
    return failure();
  SmallVector<cim::CIMTransactionInfo> transactions =
      cim::analyzeCIMTransactions(function);
  if (transactions.size() != 1) {
    function.emitError("compile-cim-transaction expects exactly one CIM "
                       "transaction, but got ")
        << transactions.size();
    return failure();
  }
  if (failed(cimframe::verifyCIMFrameModule(module)))
    return failure();

  auto profile = function->getAttrOfType<StringAttr>("cim.target_profile");
  auto profileVersion =
      function->getAttrOfType<IntegerAttr>("cim.target_profile_version");
  auto schema =
      function->getAttrOfType<IntegerAttr>("cim.execution_plan_schema_version");
  auto routePolicy = function->getAttrOfType<StringAttr>("cim.route_policy");
  if (!profile || profile.getValue() != kProfileId || !profileVersion ||
      !profileVersion.getType().isSignlessInteger(64) ||
      profileVersion.getInt() != kProfileVersion || !schema ||
      !schema.getType().isSignlessInteger(64) ||
      schema.getInt() != kExecutionPlanVersion || !routePolicy ||
      routePolicy.getValue() != kOnecastRoutePolicy) {
    function.emitError(
        "compile-cim-executable requires the verified cim22-4x5-v1 "
        "target profile and lower-left onecast policy");
    return failure();
  }

  SmallVector<uint64_t> flits;
  bool hasPacket = false;
  std::vector<StaticWeightSection> weights;
  std::vector<CIMFramePacket> packets;
  for (Operation &op : module.getBody()->getOperations()) {
    if (auto control = dyn_cast<cimframe::ControlInt8PacketOp>(op)) {
      CIMFramePacket packet{CIMPacketKind::Control};
      for (auto [index, value] : llvm::enumerate(control.getRoute()))
        packet.route[index] = value;
      packet.macroSlot = control.getMacro();
      if (auto planBinding = control->getAttrOfType<DictionaryAttr>(
              cim::CIMDialect::getPlanBindingAttrName())) {
        if (auto group = planBinding.getAs<IntegerAttr>("group_id"))
          packet.groupId = group.getInt();
        if (auto work = planBinding.getAs<IntegerAttr>("work_id"))
          packet.workId = work.getInt();
      }
      packets.push_back(packet);
      hasPacket = true;
      continue;
    }
    if (auto work = dyn_cast<cimframe::WorkOncePacketOp>(op)) {
      CIMFramePacket packet{CIMPacketKind::Work};
      for (auto [index, value] : llvm::enumerate(work.getRoute()))
        packet.route[index] = value;
      packets.push_back(packet);
      hasPacket = true;
      continue;
    }
    if (auto packet = dyn_cast<cimframe::CIMInt8WeightPacketOp>(op)) {
      FailureOr<StaticWeightSection> weight = buildStaticWeight(packet);
      if (failed(weight))
        return failure();
      auto route = packet.getRoute();
      for (auto [index, value] : llvm::enumerate(route))
        (*weight).route[index] = value;
      weights.push_back(std::move(*weight));
      packets.push_back(CIMFramePacket{CIMPacketKind::Weight});
      packets.back().route = weights.back().route;
      packets.back().groupId = weights.back().groupId;
      packets.back().workId = weights.back().workId;
      packets.back().macroSlot = weights.back().macroSlot;
      hasPacket = true;
      continue;
    }
    if (isa<cimframe::WriteInputCacheInt8PacketOp,
            cimframe::ConfigureTestReturnRoutePacketOp,
            cimframe::ReadOutputCacheInt8PacketOp>(op)) {
      op.emitOpError(
          "compile-cim-executable consumes these as semantic bindings; "
          "do not mix explicit target packet input with an execution plan");
      return failure();
    }
    if (isa<cimframe::StartInt8OnceOp, cimframe::WriteInt8WeightsOp>(op)) {
      op.emitOpError("compile-cim-executable requires packet-stage input, "
                     "not commands");
      return failure();
    }
    if (op.getName().getDialectNamespace() ==
        cimframe::CIMFrameDialect::getDialectNamespace()) {
      op.emitOpError("compile-cim-executable does not recognize this "
                     "CIMFrame operation");
      return failure();
    }
  }
  if (!hasPacket || weights.empty()) {
    module.emitError("compile-cim-executable requires a non-empty materialized "
                     "static packet stage");
    return failure();
  }
  auto encoded = encodeCIMFrameInt8Packets(module);
  if (failed(encoded))
    return failure();
  flits = std::move(*encoded);

  std::vector<CIMGroup> groups;
  std::vector<DynamicInputBinding> inputs;
  std::vector<ReadbackBinding> readbacks;
  for (Operation &container : function.getBody().front()) {
    auto transaction = dyn_cast<cim::TransactionOp>(container);
    if (!transaction)
      continue;
    for (Operation &op : transaction.getBody().front().without_terminator()) {
    if (auto input = dyn_cast<cim::ConfigureInputOp>(op)) {
      auto group = readI64(input, "group_id");
      auto work = readI64(input, "work_id");
      auto macro = readI64(input, "macro_idx");
      if (failed(group) || failed(work) || failed(macro))
        return failure();
      auto type = dyn_cast<RankedTensorType>(input.getInput().getType());
      if (!type || type.getRank() != 1 || type.getDimSize(0) != 64 ||
          !type.getElementType().isSignlessInteger(8)) {
        input.emitError(
            "CIM executable requires dynamic INT8 input tensor<64xi8>");
        return failure();
      }
      auto route = readRoute(input, "cim.mapping");
      if (failed(route))
        return failure();
      CIMGroup *groupPtr = nullptr;
      auto groupIt = findGroup(groups, *group);
      if (failed(groupIt)) {
        groups.push_back(CIMGroup{*group, {}});
        groupPtr = &groups.back();
      } else {
        groupPtr = *groupIt;
      }
      if (succeeded(findWork(*groupPtr, *work))) {
        input.emitError("CIM executable rejects duplicate work binding");
        return failure();
      }
      auto core = readI64(input, "core_idx");
      if (failed(core))
        return failure();
      if (*core >= 20 || *macro > 1)
        return input.emitError("CIM executable requires core_idx in [0, 19] "
                               "and macro_idx in [0, 1]");
      groupPtr->works.push_back(CIMWork{*work, *core, *macro, *route});
      int64_t inputSlot = *work;
      if (auto attr = input->getAttrOfType<IntegerAttr>("input_slot")) {
        if (!attr.getType().isSignlessInteger(64) || attr.getInt() < 0)
          return input.emitError("CIM executable requires input_slot >= 0");
        inputSlot = attr.getInt();
      }
      inputs.push_back(DynamicInputBinding{*group, *work, *macro, inputSlot});
      packets.push_back(CIMFramePacket{CIMPacketKind::InputCacheWrite});
      packets.back().route = *route;
      packets.back().groupId = *group;
      packets.back().workId = *work;
      packets.back().macroSlot = *macro;
      continue;
    }
    if (auto readback = dyn_cast<cim::ReadbackOp>(op)) {
      auto group = readI64(readback, "group_id");
      auto work = readI64(readback, "work_id");
      auto macro = readI64(readback, "macro_idx");
      if (failed(group) || failed(work) || failed(macro))
        return failure();
      if (*macro > 1)
        return readback.emitError(
            "CIM executable requires macro_idx in [0, 1]");
      auto type = dyn_cast<RankedTensorType>(readback.getResult().getType());
      if (!type || type.getRank() != 1 || type.getDimSize(0) != 16 ||
          !type.getElementType().isSignlessInteger(21)) {
        readback.emitError("CIM executable requires readback tensor<16xi21>");
        return failure();
      }
      auto route = readRoute(readback, "cim.mapping");
      auto testCore = readTestCore(readback.getOperation());
      auto address = readOutputAddress(readback);
      if (failed(route) || failed(testCore) || failed(address))
        return failure();
      auto groupIt = findGroup(groups, *group);
      if (failed(groupIt) || failed(findWork(**groupIt, *work))) {
        readback.emitError("CIM executable readback references unknown work");
        return failure();
      }
      readbacks.push_back(
          ReadbackBinding{*group, *work, *macro, *address, *route, *testCore});
      packets.push_back(CIMFramePacket{CIMPacketKind::ReturnRoute});
      packets.back().route = *route;
      packets.back().groupId = *group;
      packets.back().workId = *work;
      packets.back().macroSlot = *macro;
      packets.back().testCore = *testCore;
      packets.push_back(CIMFramePacket{CIMPacketKind::Control});
      packets.back().route = *route;
      packets.back().groupId = *group;
      packets.back().workId = *work;
      packets.back().macroSlot = *macro;
      packets.push_back(CIMFramePacket{CIMPacketKind::OutputCacheRead});
      packets.back().route = *route;
      packets.back().groupId = *group;
      packets.back().workId = *work;
      packets.back().macroSlot = *macro;
      packets.back().cacheAddress = *address;
      continue;
    }
    if (auto once = dyn_cast<cim::OnceOp>(op)) {
      auto group = readI64(once, "group_id");
      auto route = readRoute(once, "cim.mapping");
      if (failed(group) || failed(route))
        return failure();
      CIMFramePacket packet{CIMPacketKind::Work};
      packet.route = *route;
      packet.groupId = *group;
      packets.push_back(packet);
    }
    }
  }
  if (groups.empty() || inputs.empty() || readbacks.empty()) {
    function.emitError(
        "compile-cim-executable requires input and readback bindings");
    return failure();
  }
  for (const CIMGroup &group : groups)
    for (const CIMWork &work : group.works)
      if (failed(findWeight(weights, group.groupId, work.workId))) {
        function.emitError(
            "CIM executable is missing a static weight for work ")
            << work.workId;
        return failure();
      }

  return CIMTransaction(profile.getValue().str(), profileVersion.getInt(),
                       schema.getInt(), std::move(groups), std::move(weights),
                       std::move(packets), std::move(inputs),
                       std::move(readbacks),
                       std::vector<uint64_t>(flits.begin(), flits.end()));
}

} // namespace mlir::cim22::target
