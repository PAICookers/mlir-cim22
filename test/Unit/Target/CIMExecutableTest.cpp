//===- CIMTransactionTest.cpp - CIM22 target plan tests -----*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIM22/Target/CIMExecutable.h"
#include "CIM22/Conversion/LinalgToCIM/CIMSegments.h"
#include "CIM22/Dialect/CIM/IR/CIMDialect.h"
#include "CIM22/Dialect/CIM/IR/CIMOps.h"
#include "CIM22/Dialect/CIMFrame/IR/CIMFrameDialect.h"
#include "CIM22/Dialect/CIMFrame/IR/CIMFrameOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <limits>

using namespace mlir;
using namespace mlir::cimframe;
using mlir::cim22::target::compileCIMTransaction;

static bool check(bool condition, const char *message) {
  if (condition)
    return true;
  std::cerr << "FAIL: " << message << '\n';
  return false;
}

static OwningOpRef<ModuleOp> parsePlan(MLIRContext &context) {
  return parseSourceString<ModuleOp>(R"mlir(
module {
  cim.static_weight @weight = dense<0> : tensor<16x64xi8>
  func.func @invoke(%input: tensor<64xi8>) attributes {
      cim.execution_plan_schema_version = 1 : i64,
      cim.placement_policy = "core-major-dual-macro-v1",
      cim.route_policy = "lower-left-maximal-xy-v1",
      cim.target_profile = "cim22-4x5-v1",
      cim.target_profile_version = 1 : i64} {
    %transaction = "cim.transaction"(%input) ({
    ^bb0(%transaction_input: tensor<64xi8>):
      cim.configure_input %transaction_input {cim.mapping = {core_coord = array<i64: 0, 0>, destination = array<i64: 0, 0>, ingress = array<i64: 0, 0>, route = array<i64: 0, 0, 0, 0, 0, 0>, source = array<i64: 0, 0>}, core_idx = 0 : i64, group_id = 0 : i64, k_tile = 0 : i64, m_tile = 0 : i64, macro_idx = 0 : i64, n_tile = 0 : i64, work_id = 0 : i64} : tensor<64xi8>
      cim.configure_weight @weight {cim.mapping = {core_coord = array<i64: 0, 0>, destination = array<i64: 0, 0>, ingress = array<i64: 0, 0>, route = array<i64: 0, 0, 0, 0, 0, 0>, source = array<i64: 0, 0>}, core_idx = 0 : i64, group_id = 0 : i64, k_tile = 0 : i64, m_tile = 0 : i64, macro_idx = 0 : i64, n_tile = 0 : i64, work_id = 0 : i64}
      cim.dispatch {cim.mapping = {core_coord = array<i64: 0, 0>, destination = array<i64: 0, 0>, ingress = array<i64: 0, 0>, route = array<i64: 0, 0, 0, 0, 0, 0>, source = array<i64: 0, 0>}, core_idx = 0 : i64, group_id = 0 : i64, k_tile = 0 : i64, m_tile = 0 : i64, macro_idx = 0 : i64, n_tile = 0 : i64, work_id = 0 : i64}
      cim.once {cim.mapping = {core_coord = array<i64: 0, 0>, destination = array<i64: 0, 0>, ingress = array<i64: 0, 0>, route = array<i64: 0, 0, 0, 0, 0, 0>, source = array<i64: 0, 0>}, core_idx = 0 : i64, group_id = 0 : i64}
      %read = cim.readback {cim.mapping = {core_coord = array<i64: 0, 0>, destination = array<i64: 0, 0>, ingress = array<i64: 0, 0>, route = array<i64: 0, 0, 0, 0, 0, 0>, source = array<i64: 0, 0>}, core_idx = 0 : i64, group_id = 0 : i64, k_tile = 0 : i64, m_tile = 0 : i64, macro_idx = 0 : i64, n_tile = 0 : i64, output_cache_address = 7 : i64, test_core_xy = 0 : i64, test_core_x = 0 : i64, test_core_y = 0 : i64, work_id = 0 : i64} : tensor<16xi21>
      cim.group_barrier {group_id = 0 : i64}
      "cim.yield"(%read) : (tensor<16xi21>) -> ()
    }) {cim.transaction_idx = 0 : i64} : (tensor<64xi8>) -> tensor<16xi21>
    return
  }
}
)mlir",
                                     &context);
}

static bool addPacketStage(ModuleOp module, int64_t workId = 0) {
  OpBuilder builder(module.getContext());
  builder.setInsertionPointToStart(module.getBody());
  const std::array<int32_t, 6> route{};
  auto routeAttr = builder.getDenseI32ArrayAttr(route);
  auto control =
      ControlInt8PacketOp::create(builder, builder.getUnknownLoc(), routeAttr,
                                  builder.getI32IntegerAttr(workId));

  SmallVector<int32_t, 256> words(256, 0);
  SmallVector<NamedAttribute> planBinding;
  planBinding.push_back(builder.getNamedAttr(
      "function", FlatSymbolRefAttr::get(builder.getContext(), "invoke")));
  planBinding.push_back(builder.getNamedAttr(
      "resource", FlatSymbolRefAttr::get(builder.getContext(), "weight")));
  planBinding.push_back(
      builder.getNamedAttr("group_id", builder.getI64IntegerAttr(0)));
  planBinding.push_back(
      builder.getNamedAttr("work_id", builder.getI64IntegerAttr(workId)));
  planBinding.push_back(
      builder.getNamedAttr("core_idx", builder.getI64IntegerAttr(0)));
  planBinding.push_back(
      builder.getNamedAttr("macro_idx", builder.getI64IntegerAttr(workId)));
  auto mapping = builder.getDictionaryAttr({builder.getNamedAttr(
      "route", builder.getDenseI64ArrayAttr({0, 0, 0, 0, 0, 0}))});
  planBinding.push_back(builder.getNamedAttr("mapping", mapping));
  auto weight = CIMInt8WeightPacketOp::create(
      builder, builder.getUnknownLoc(), routeAttr,
      DenseIntElementsAttr::get(
          RankedTensorType::get({256}, builder.getI32Type()), words));
  weight->setAttr("cim.plan_binding", builder.getDictionaryAttr(planBinding));
  return control && weight;
}

static OwningOpRef<ModuleOp> parseTwoWorkPlan(MLIRContext &context,
                                              bool sharedInput) {
  auto module = parsePlan(context);
  if (!module)
    return {};
  func::FuncOp function = *module->getOps<func::FuncOp>().begin();
  auto original =
      *function.getBody().front().getOps<cim::TransactionOp>().begin();
  Block &originalBody = original.getBody().front();
  auto input = *originalBody.getOps<cim::ConfigureInputOp>().begin();
  auto weight = *originalBody.getOps<cim::ConfigureWeightOp>().begin();
  auto dispatch = *originalBody.getOps<cim::DispatchOp>().begin();
  auto once = *originalBody.getOps<cim::OnceOp>().begin();
  auto readback = *originalBody.getOps<cim::ReadbackOp>().begin();
  auto barrier = *originalBody.getOps<cim::GroupBarrierOp>().begin();
  Location location = original.getLoc();
  if (failed(
          function.insertArgument(1, input.getInput().getType(), {}, location)))
    return {};

  // Reverse the transaction operands so its block argument ordinal differs
  // from the Host ABI slot. Shared-input works both use the second Host input.
  SmallVector<Value> operands{function.getArgument(1)};
  if (!sharedInput)
    operands.push_back(function.getArgument(0));
  OpBuilder builder(original);
  SmallVector<Type> resultTypes(2, readback.getResult().getType());
  auto transaction = cim::TransactionOp::create(builder, location, resultTypes,
                                                operands, original->getAttrs());
  Block &body = transaction.getBody().emplaceBlock();
  for (Value operand : operands)
    body.addArgument(operand.getType(), location);
  builder.setInsertionPointToEnd(&body);
  auto cloneWork = [&](Operation *op, int64_t workId) {
    IRMapping mapping;
    mapping.map(originalBody.getArgument(0),
                body.getArgument(sharedInput ? 0 : workId));
    Operation *clone = builder.clone(*op, mapping);
    clone->setAttr("work_id", builder.getI64IntegerAttr(workId));
    clone->setAttr("macro_idx", builder.getI64IntegerAttr(workId));
    return clone;
  };
  for (int64_t workId = 0; workId < 2; ++workId) {
    cloneWork(input, workId)
        ->setAttr("input_cache_address",
                  builder.getI64IntegerAttr(workId == 0 ? 3 : 7));
    cloneWork(weight, workId);
  }
  for (int64_t workId = 0; workId < 2; ++workId)
    cloneWork(dispatch, workId);
  builder.clone(*once.getOperation());
  SmallVector<Value> results;
  for (int64_t workId = 0; workId < 2; ++workId)
    results.push_back(cloneWork(readback, workId)->getResult(0));
  builder.clone(*barrier.getOperation());
  cim::YieldOp::create(builder, location, results);
  original.erase();
  if (!addPacketStage(*module) || !addPacketStage(*module, 1))
    return {};
  return module;
}

static bool addBF16PacketStage(ModuleOp module) {
  OpBuilder builder(module.getContext());
  builder.setInsertionPointToStart(module.getBody());
  const std::array<int32_t, 6> route{};
  auto routeAttr = builder.getDenseI32ArrayAttr(route);
  auto control =
      ControlBF16PacketOp::create(builder, builder.getUnknownLoc(), routeAttr,
                                  builder.getI32IntegerAttr(0));

  SmallVector<NamedAttribute> planBinding;
  planBinding.push_back(builder.getNamedAttr(
      "function", FlatSymbolRefAttr::get(builder.getContext(), "invoke")));
  planBinding.push_back(builder.getNamedAttr(
      "resource", FlatSymbolRefAttr::get(builder.getContext(), "weight")));
  for (StringRef name : {"group_id", "work_id", "core_idx", "macro_idx"})
    planBinding.push_back(
        builder.getNamedAttr(name, builder.getI64IntegerAttr(0)));
  auto mapping = builder.getDictionaryAttr({builder.getNamedAttr(
      "route", builder.getDenseI64ArrayAttr({0, 0, 0, 0, 0, 0}))});
  planBinding.push_back(builder.getNamedAttr("mapping", mapping));
  DictionaryAttr binding = builder.getDictionaryAttr(planBinding);

  constexpr std::array<int64_t, 2> exponents{INT64_C(0x0706050403020100),
                                             INT64_C(0x0f0e0d0c0b0a0908)};
  auto exponent = WeightExponentPacketOp::create(
      builder, builder.getUnknownLoc(), routeAttr,
      builder.getDenseI64ArrayAttr(exponents));
  SmallVector<int32_t, 256> words(256, 0);
  auto weight = CIMBF16WeightPacketOp::create(
      builder, builder.getUnknownLoc(), routeAttr,
      DenseIntElementsAttr::get(
          RankedTensorType::get({256}, builder.getI32Type()), words));
  for (Operation *op :
       {control.getOperation(), exponent.getOperation(), weight.getOperation()})
    op->setAttr("cim.plan_binding", binding);
  return control && exponent && weight;
}

static bool testConstruction(MLIRContext &context) {
  auto module = parsePlan(context);
  if (!check(static_cast<bool>(module), "hardware plan fixture parses") ||
      !check(addPacketStage(*module), "packet stage is materialized"))
    return false;

  func::FuncOp function = *module->getOps<func::FuncOp>().begin();
  auto transactions = cim::analyzeCIMTransactions(function);
  if (!check(transactions.size() == 1 &&
                 transactions.front().inputs.size() == 1 &&
                 transactions.front().outputs.size() == 1,
             "transaction analysis preserves typed SSA boundaries"))
    return false;

  auto executable = compileCIMTransaction(*module);
  if (!check(succeeded(executable), "CIM executable construction succeeds"))
    return false;
  const auto &value = *executable;
  return check(value.getTargetProfile() == "cim22-4x5-v1",
               "target profile is preserved") &&
         check(value.getTargetProfileVersion() == 1,
               "target profile version is preserved") &&
         check(value.getExecutionPlanVersion() == 1,
               "execution plan version is preserved") &&
         check(value.getGroups().size() == 1 &&
                   value.getGroups()[0].works.size() == 1,
               "group and work table is materialized") &&
         check(value.getStaticWeights().size() == 1 &&
                   value.getStaticWeights()[0].words.size() == 256,
               "static weight section is materialized") &&
         check(
             value.getDynamicInputs().size() == 1 &&
                 value.getDynamicInputs()[0].inputSlot == 0 &&
                 value.getDynamicInputs()[0].inputCacheAddress == 0,
             "dynamic input binding keeps input slot and default Cache row") &&
         check(value.getReadbacks().size() == 1 &&
                   value.getReadbacks()[0].outputCacheAddress == 7,
               "readback binding keeps output Cache address") &&
         check(value.getPackets().size() == 7,
               "semantic packets preserve input and readback order") &&
         check(value.getFlits().size() == 258,
               "static packet flits are consumed without remapping");
}

static bool testInputArgumentMapping(MLIRContext &context) {
  for (bool sharedInput : {false, true}) {
    auto module = parseTwoWorkPlan(context, sharedInput);
    if (!check(static_cast<bool>(module), "two-work input fixture is built"))
      return false;
    auto executable = compileCIMTransaction(*module);
    if (!check(succeeded(executable), "two-work input fixture compiles"))
      return false;
    auto inputs = executable->getDynamicInputs();
    if (!check(inputs.size() == 2, "both works have dynamic input bindings") ||
        !check(inputs[0].inputSlot == 1 &&
                   inputs[1].inputSlot == (sharedInput ? 1 : 0),
               sharedInput
                   ? "shared transaction input keeps one Host ABI slot"
                   : "transaction operands map to Host argument slots") ||
        !check(inputs[0].inputCacheAddress == 3 &&
                   inputs[1].inputCacheAddress == 7,
               "nonzero input Cache rows survive target compilation"))
      return false;
  }
  return true;
}

static bool testRejectsInvalidInputCacheAddress(MLIRContext &context) {
  Builder builder(&context);
  const std::array<IntegerAttr, 3> addresses{builder.getI64IntegerAttr(-1),
                                             builder.getI64IntegerAttr(8),
                                             builder.getI32IntegerAttr(3)};
  for (IntegerAttr address : addresses) {
    auto module = parsePlan(context);
    if (!check(static_cast<bool>(module), "invalid Cache fixture parses") ||
        !check(addPacketStage(*module), "invalid Cache packet stage is built"))
      return false;
    module->walk([&](cim::ConfigureInputOp input) {
      input->setAttr("input_cache_address", address);
    });
    ScopedDiagnosticHandler diagnostics(&context, [](Diagnostic &) {});
    if (!check(failed(compileCIMTransaction(*module)),
               "input Cache row requires i64 in [0, 7]"))
      return false;
  }
  return true;
}

static bool testRejectsMultipleTransactions(MLIRContext &context) {
  auto module = parsePlan(context);
  if (!check(static_cast<bool>(module), "multi-transaction fixture parses") ||
      !check(addPacketStage(*module),
             "multi-transaction packet stage is materialized"))
    return false;

  func::FuncOp function = *module->getOps<func::FuncOp>().begin();
  Block &body = function.getBody().front();
  Operation *transaction = nullptr;
  for (Operation &op : body)
    if (isa<cim::TransactionOp>(op))
      transaction = &op;
  if (!check(transaction, "multi-transaction fixture has a transaction"))
    return false;

  OpBuilder builder(body.getTerminator());
  IRMapping mapping;
  mapping.map(body.getArgument(0), body.getArgument(0));
  Operation *clone = builder.clone(*transaction, mapping);
  clone->setAttr("cim.transaction_idx", builder.getI64IntegerAttr(1));

  auto transactions = cim::analyzeCIMTransactions(function);
  if (!check(transactions.size() == 2 && transactions[0].transactionIdx == 0 &&
                 transactions[1].transactionIdx == 1 &&
                 transactions[0].inputs.size() == 1 &&
                 transactions[0].outputs.size() == 1 &&
                 transactions[1].inputs.size() == 1 &&
                 transactions[1].outputs.size() == 1,
             "transaction analysis preserves ordered launch boundaries"))
    return false;

  ScopedDiagnosticHandler diagnostics(&context, [](Diagnostic &) {});
  return check(
      failed(compileCIMTransaction(*module)),
      "multiple CIM transactions are rejected before frame compilation");
}

static bool testRejectsMissingReadbackBinding(MLIRContext &context) {
  auto module = parsePlan(context);
  if (!check(static_cast<bool>(module), "negative fixture parses") ||
      !check(addPacketStage(*module), "negative packet stage is materialized"))
    return false;
  bool removed = false;
  module->walk([&](cim::ReadbackOp readback) {
    readback->removeAttr("output_cache_address");
    removed = true;
  });
  if (!check(removed, "negative fixture has a readback"))
    return false;
  ScopedDiagnosticHandler diagnostics(&context, [](Diagnostic &) {});
  return check(failed(compileCIMTransaction(*module)),
               "missing output Cache binding is rejected");
}

static bool testBF16Construction(MLIRContext &context) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
module {
  cim.static_weight @weight = dense<0.0> : tensor<16x64xbf16>
  func.func @invoke(%input: tensor<64xbf16>) attributes {
      cim.execution_plan_schema_version = 1 : i64,
      cim.placement_policy = "core-major-dual-macro-v1",
      cim.route_policy = "lower-left-maximal-xy-v1",
      cim.target_profile = "cim22-4x5-v1",
      cim.target_profile_version = 1 : i64} {
    %transaction = "cim.transaction"(%input) ({
    ^bb0(%transaction_input: tensor<64xbf16>):
      cim.configure_input %transaction_input {cim.mapping = {route = array<i64: 0, 0, 0, 0, 0, 0>}, core_idx = 0 : i64, group_id = 0 : i64, k_tile = 0 : i64, m_tile = 0 : i64, macro_idx = 0 : i64, n_tile = 0 : i64, work_id = 0 : i64} : tensor<64xbf16>
      cim.configure_weight @weight {cim.mapping = {route = array<i64: 0, 0, 0, 0, 0, 0>}, core_idx = 0 : i64, group_id = 0 : i64, k_tile = 0 : i64, m_tile = 0 : i64, macro_idx = 0 : i64, n_tile = 0 : i64, work_id = 0 : i64}
      cim.dispatch {cim.mapping = {route = array<i64: 0, 0, 0, 0, 0, 0>}, core_idx = 0 : i64, group_id = 0 : i64, k_tile = 0 : i64, m_tile = 0 : i64, macro_idx = 0 : i64, n_tile = 0 : i64, work_id = 0 : i64}
      cim.once {cim.mapping = {route = array<i64: 0, 0, 0, 0, 0, 0>}, core_idx = 0 : i64, group_id = 0 : i64}
      %read = cim.readback {cim.mapping = {route = array<i64: 0, 0, 0, 0, 0, 0>}, core_idx = 0 : i64, group_id = 0 : i64, k_tile = 0 : i64, m_tile = 0 : i64, macro_idx = 0 : i64, n_tile = 0 : i64, output_cache_address = 7 : i64, test_core_xy = 0 : i64, test_core_x = 0 : i64, test_core_y = 0 : i64, work_id = 0 : i64} : tensor<16xbf16>
      cim.group_barrier {group_id = 0 : i64}
      "cim.yield"(%read) : (tensor<16xbf16>) -> ()
    }) {cim.transaction_idx = 0 : i64} : (tensor<64xbf16>) -> tensor<16xbf16>
    return
  }
}
)mlir",
                                            &context);
  if (!check(static_cast<bool>(module), "BF16 plan fixture parses"))
    return false;
  if (!check(addBF16PacketStage(*module), "BF16 packet stage is materialized"))
    return false;
  auto executable = compileCIMTransaction(*module);
  if (!check(succeeded(executable), "BF16 executable construction succeeds"))
    return false;
  const auto &value = *executable;
  const auto &weight = value.getStaticWeights().front();
  return check(weight.dataType == ::cim22::execution::CIMDataType::BF16,
               "BF16 weight mode is preserved") &&
         check(weight.exponents.front() == 0 && weight.exponents.back() == 15,
               "Weight_EXP lanes are unpacked little-endian") &&
         check(value.getDynamicInputs().front().dataType ==
                       ::cim22::execution::CIMDataType::BF16 &&
                   value.getReadbacks().front().dataType ==
                       ::cim22::execution::CIMDataType::BF16,
               "BF16 dynamic bindings preserve their mode") &&
         check(value.getPackets().size() == 8,
               "BF16 semantic packets include Weight_EXP") &&
         check(value.getFlits().size() == 261,
               "BF16 static packet flits include Weight_EXP body");
}

int main() {
  MLIRContext context;
  context.loadDialect<cim::CIMDialect, CIMFrameDialect, arith::ArithDialect,
                      func::FuncDialect, tensor::TensorDialect>();
  return testConstruction(context) &&
                 testRejectsMultipleTransactions(context) &&
                 testInputArgumentMapping(context) &&
                 testRejectsInvalidInputCacheAddress(context) &&
                 testRejectsMissingReadbackBinding(context) &&
                 testBF16Construction(context)
             ? 0
             : 1;
}
