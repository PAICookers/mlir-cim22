//===- Passes.cpp - CIMFrame passes ----------------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIM22/Dialect/CIMFrame/Transforms/Passes.h"

#include "CIM22/Dialect/CIM/IR/CIMOps.h"
#include "CIM22/Dialect/CIMFrame/IR/CIMFrameDialect.h"
#include "CIM22/Support/Int8WeightLayout.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/bit.h"

#include <array>
#include <cstdint>

namespace mlir::cimframe {
#define GEN_PASS_DEF_MATERIALIZECIMSTATICWEIGHTSECTION
#define GEN_PASS_DEF_LOWERCIMFRAMECOMMANDSTOPACKETS
#define GEN_PASS_DEF_VERIFYCIMFRAME
#include "CIM22/Dialect/CIMFrame/Transforms/Passes.h.inc"

namespace {
constexpr llvm::StringLiteral kProvenanceAttrs[] = {
    "m_tile",   "n_tile",    "k_tile",     "work_id",
    "group_id", "core_slot", "macro_slot", "cim.mapping"};

struct StaticWeightCommandPlan {
  Location location;
  DenseI32ArrayAttr route;
  IntegerAttr macro;
  DenseI32ArrayAttr words;
  DictionaryAttr provenance;
};

LogicalResult verifyExactFunctionContract(func::FuncOp function) {
  auto requireI64 = [&](StringRef name, int64_t expected) {
    auto value = function->getAttrOfType<IntegerAttr>(name);
    if (!value || !value.getType().isSignlessInteger(64) ||
        value.getInt() != expected) {
      function.emitError("materialize-cim-static-weight-section requires ")
          << name << " = " << expected << " : i64";
      return failure();
    }
    return success();
  };
  auto requireString = [&](StringRef name, StringRef expected) {
    auto value = function->getAttrOfType<StringAttr>(name);
    if (!value || value.getValue() != expected) {
      function.emitError("materialize-cim-static-weight-section requires ")
          << name << " = '" << expected << "'";
      return failure();
    }
    return success();
  };

  return success(
      succeeded(requireI64("cim.execution_plan_schema_version", 1)) &&
      succeeded(requireString("cim.target_profile", "cim22-4x5-v1")) &&
      succeeded(requireI64("cim.target_profile_version", 1)) &&
      succeeded(
          requireString("cim.placement_policy", "core-major-dual-macro-v1")) &&
      succeeded(requireString("cim.route_policy", "lower-left-maximal-xy-v1")));
}

FailureOr<StaticWeightCommandPlan>
planStaticWeightCommand(cim::ConfigureWeightOp op, func::FuncOp function,
                        OpBuilder &builder) {
  for (NamedAttribute attr : op->getAttrs()) {
    StringRef name = attr.getName().strref();
    if (name != "resource" && !llvm::is_contained(kProvenanceAttrs, name)) {
      op.emitOpError("materialize-cim-static-weight-section rejects unexpected "
                     "configure_weight attribute '")
          << name << "'";
      return failure();
    }
  }

  auto resource = dyn_cast<FlatSymbolRefAttr>(op.getResource());
  if (!resource) {
    op.emitOpError("materialize-cim-static-weight-section requires a flat "
                   "same-module static weight reference");
    return failure();
  }
  cim::StaticWeightOp weight =
      SymbolTable::lookupNearestSymbolFrom<cim::StaticWeightOp>(op, resource);

  SmallVector<IntegerAttr> integers;
  integers.reserve(7);
  for (StringRef name : ArrayRef(kProvenanceAttrs).drop_back())
    integers.push_back(op->getAttrOfType<IntegerAttr>(name));
  IntegerAttr macroSlot = integers.back();

  auto mapping = cast<DictionaryAttr>(op->getAttr("cim.mapping"));
  auto sourceRoute = cast<DenseI64ArrayAttr>(mapping.get("route"));
  SmallVector<int32_t, 6> route;
  llvm::transform(sourceRoute.asArrayRef(), std::back_inserter(route),
                  [](int64_t value) { return static_cast<int32_t>(value); });

  std::array<uint8_t, 16 * 64> bytes{};
  for (auto [index, value] :
       llvm::enumerate(weight.getValue().getValues<APInt>()))
    bytes[index] = static_cast<uint8_t>(value.getZExtValue());

  // HWSRC-046 accepts the logical tile mapping exercised by
  // materialize-static-weight-section.mlir.
  // FIXME(CTQ-020): That test remains software-only and is not board
  // verification.
  std::array<uint32_t, 256> rawWords =
      cim22::mapInt8WeightTileToCIMWords(bytes);
  SmallVector<int32_t, 256> words;
  llvm::transform(rawWords, std::back_inserter(words),
                  [](uint32_t word) { return llvm::bit_cast<int32_t>(word); });

  SmallVector<NamedAttribute> provenance;
  provenance.push_back(builder.getNamedAttr(
      "function",
      FlatSymbolRefAttr::get(function.getContext(), function.getSymName())));
  provenance.push_back(builder.getNamedAttr("resource", resource));
  for (auto [name, value] :
       llvm::zip(ArrayRef(kProvenanceAttrs).drop_back(), integers))
    provenance.push_back(builder.getNamedAttr(name, value));
  provenance.push_back(builder.getNamedAttr("mapping", mapping));

  return StaticWeightCommandPlan{
      op.getLoc(), builder.getDenseI32ArrayAttr(route),
      builder.getI32IntegerAttr(static_cast<int32_t>(macroSlot.getInt())),
      builder.getDenseI32ArrayAttr(words),
      builder.getDictionaryAttr(provenance)};
}

class MaterializeCIMStaticWeightSection final
    : public impl::MaterializeCIMStaticWeightSectionBase<
          MaterializeCIMStaticWeightSection> {
public:
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    for (Operation &op : module.getBody()->getOperations()) {
      if (op.getName().getDialectNamespace() ==
          CIMFrameDialect::getDialectNamespace()) {
        module.emitError("materialize-cim-static-weight-section rejects "
                         "pre-existing top-level cimframe operations");
        return signalPassFailure();
      }
    }

    OpBuilder builder(&getContext());
    SmallVector<StaticWeightCommandPlan> plans;
    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
      SmallVector<cim::ConfigureWeightOp> configureWeights;
      function.walk(
          [&](cim::ConfigureWeightOp op) { configureWeights.push_back(op); });
      if (configureWeights.empty())
        continue;
      if (failed(verifyExactFunctionContract(function))) {
        signalPassFailure();
        return;
      }
      for (cim::ConfigureWeightOp op : configureWeights) {
        FailureOr<StaticWeightCommandPlan> plan =
            planStaticWeightCommand(op, function, builder);
        if (failed(plan)) {
          signalPassFailure();
          return;
        }
        plans.push_back(*plan);
      }
    }
    if (plans.empty())
      return;

    auto firstFunction = *module.getOps<func::FuncOp>().begin();
    builder.setInsertionPoint(firstFunction);
    for (const StaticWeightCommandPlan &plan : plans) {
      WriteInt8WeightsOp command = WriteInt8WeightsOp::create(
          builder, plan.location, plan.route, plan.macro, plan.words);
      command->setAttr("cim.provenance", plan.provenance);
    }
  }
};

class LowerStartInt8Once final : public OpConversionPattern<StartInt8OnceOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StartInt8OnceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ControlInt8PacketOp::create(rewriter, op.getLoc(), adaptor.getRoute(),
                                adaptor.getMacro());
    WorkOncePacketOp::create(rewriter, op.getLoc(), adaptor.getRoute());
    rewriter.eraseOp(op);
    return success();
  }
};

class LowerWriteInt8Weights final
    : public OpConversionPattern<WriteInt8WeightsOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(WriteInt8WeightsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ControlInt8PacketOp control = ControlInt8PacketOp::create(
        rewriter, op.getLoc(), adaptor.getRoute(), adaptor.getMacro());
    CIMInt8WeightPacketOp weight = CIMInt8WeightPacketOp::create(
        rewriter, op.getLoc(), adaptor.getRoute(), adaptor.getWords());
    if (Attribute provenance = op->getAttr("cim.provenance")) {
      control->setAttr("cim.provenance", provenance);
      weight->setAttr("cim.provenance", provenance);
    }
    rewriter.eraseOp(op);
    return success();
  }
};

class LowerCIMFrameCommandsToPackets final
    : public impl::LowerCIMFrameCommandsToPacketsBase<
          LowerCIMFrameCommandsToPackets> {
public:
  using Base::Base;

  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addIllegalDialect<CIMFrameDialect>();
    target.addLegalOp<ControlInt8PacketOp, WorkOncePacketOp,
                      CIMInt8WeightPacketOp>();
    target.markUnknownOpDynamicallyLegal([](Operation *op) {
      return op->getName().getDialectNamespace() !=
             CIMFrameDialect::getDialectNamespace();
    });

    RewritePatternSet patterns(&getContext());
    patterns.add<LowerStartInt8Once, LowerWriteInt8Weights>(&getContext());
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

class VerifyCIMFrame final : public impl::VerifyCIMFrameBase<VerifyCIMFrame> {
public:
  using Base::Base;

  void runOnOperation() override {
    Block::OpListType &operations = getOperation().getBody()->getOperations();
    Operation *firstCommand = nullptr;
    Operation *firstPacket = nullptr;
    for (Operation &op : operations) {
      if (isa<StartInt8OnceOp, WriteInt8WeightsOp>(op))
        firstCommand = firstCommand ? firstCommand : &op;
      if (isa<ControlInt8PacketOp, WorkOncePacketOp, CIMInt8WeightPacketOp>(op))
        firstPacket = firstPacket ? firstPacket : &op;
    }

    if (firstCommand && firstPacket) {
      getOperation().emitError(
          "cannot mix cimframe command and packet stages in one module");
      return signalPassFailure();
    }
    if (!firstPacket)
      return;

    bool invalid = false;
    for (Operation &op : operations) {
      if (auto control = dyn_cast<ControlInt8PacketOp>(op)) {
        Operation *next = op.getNextNode();
        if (auto work = dyn_cast_or_null<WorkOncePacketOp>(next)) {
          if (control.getRoute() != work.getRoute()) {
            work.emitOpError("expects control/work routes to match");
            invalid = true;
          }
        } else if (auto weight =
                       dyn_cast_or_null<CIMInt8WeightPacketOp>(next)) {
          if (control.getRoute() != weight.getRoute()) {
            weight.emitOpError("expects control/weight routes to match");
            invalid = true;
          }
        } else {
          control.emitOpError(
              "expects control_int8_packet immediately followed by "
              "work_once_packet or cim_int8_weight_packet");
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
      }
    }
    if (invalid)
      signalPassFailure();
  }
};
} // namespace
} // namespace mlir::cimframe
