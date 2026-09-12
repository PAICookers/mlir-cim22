//===- Passes.cpp - CIMFrame passes ----------------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIM22/Dialect/CIMFrame/Transforms/Passes.h"

#include "CIM22/Dialect/CIM/IR/CIMDialect.h"
#include "CIM22/Dialect/CIM/IR/CIMOps.h"
#include "CIM22/Dialect/CIMFrame/IR/CIMFrameDialect.h"
#include "CIM22/Support/BF16Support.h"
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
constexpr llvm::StringLiteral kPlanBindingAttrs[] = {
    "m_tile",   "n_tile",    "k_tile",     "work_id",
    "group_id", "core_idx", "macro_idx", "cim.mapping"};

struct StaticWeightCommandPlan {
  Location location;
  DenseI32ArrayAttr route;
  IntegerAttr macro;
  DenseIntElementsAttr words;
  DenseI64ArrayAttr exponentWords;
  bool bf16 = false;
  DictionaryAttr planBinding;
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
    if (name != "resource" && name != cim::CIMDialect::getTransactionIdxAttrName() &&
        !llvm::is_contained(kPlanBindingAttrs, name)) {
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
  auto int8Weight = dyn_cast<DenseIntElementsAttr>(weight.getValue());
  auto bf16Weight = dyn_cast<DenseFPElementsAttr>(weight.getValue());
  if (!int8Weight && !bf16Weight)
    return op.emitOpError("materialize-cim-static-weight-section expects an "
                          "INT8 or BF16 static weight");

  SmallVector<IntegerAttr> integers;
  integers.reserve(7);
  for (StringRef name : ArrayRef(kPlanBindingAttrs).drop_back())
    integers.push_back(op->getAttrOfType<IntegerAttr>(name));
  IntegerAttr macroSlot = integers.back();

  auto mapping = cast<DictionaryAttr>(op->getAttr("cim.mapping"));
  auto sourceRoute = cast<DenseI64ArrayAttr>(mapping.get("route"));
  SmallVector<int32_t, 6> route;
  llvm::transform(sourceRoute.asArrayRef(), std::back_inserter(route),
                  [](int64_t value) { return static_cast<int32_t>(value); });

  std::array<uint8_t, 16 * 64> bytes{};
  DenseI64ArrayAttr exponentWords;
  bool isBF16 = static_cast<bool>(bf16Weight);
  if (int8Weight) {
    for (auto [index, value] :
         llvm::enumerate(int8Weight.getValues<APInt>()))
      bytes[index] = static_cast<uint8_t>(value.getZExtValue());
  } else {
    std::array<uint16_t, 16 * 64> values{};
    for (auto [index, value] :
         llvm::enumerate(bf16Weight.getValues<APFloat>())) {
      values[index] = static_cast<uint16_t>(value.bitcastToAPInt().getZExtValue());
      if (((values[index] >> 7) & 0xff) == 0xff)
        return op.emitOpError(
            "materialize-cim-static-weight-section does not support BF16 "
            "NaN or infinity weights");
    }
    const cim22::BF16PrealignedWeightTile prealigned =
        cim22::prealignBF16WeightTile(values);
    for (auto [index, value] : llvm::enumerate(prealigned.mantissas))
      bytes[index] = static_cast<uint8_t>(value);
    const auto exponents = cim22::packBF16WeightExponents(prealigned.exponents);
    exponentWords = builder.getDenseI64ArrayAttr(
        {llvm::bit_cast<int64_t>(exponents[0]),
         llvm::bit_cast<int64_t>(exponents[1])});
  }

  // HWSRC-046 accepts the logical tile mapping exercised by
  // materialize-static-weight-section.mlir.
  // FIXME(CTQ-020): That test remains software-only and is not board
  // verification.
  std::array<uint32_t, 256> rawWords =
      cim22::mapInt8WeightTileToCIMWords(bytes);
  SmallVector<int32_t, 256> words;
  llvm::transform(rawWords, std::back_inserter(words),
                  [](uint32_t word) { return llvm::bit_cast<int32_t>(word); });

  SmallVector<NamedAttribute> planBinding;
  planBinding.push_back(builder.getNamedAttr(
      "function",
      FlatSymbolRefAttr::get(function.getContext(), function.getSymName())));
  planBinding.push_back(builder.getNamedAttr("resource", resource));
  for (auto [name, value] :
       llvm::zip(ArrayRef(kPlanBindingAttrs).drop_back(), integers))
    planBinding.push_back(builder.getNamedAttr(name, value));
  planBinding.push_back(builder.getNamedAttr("mapping", mapping));

  return StaticWeightCommandPlan{
      op.getLoc(), builder.getDenseI32ArrayAttr(route),
      builder.getI32IntegerAttr(static_cast<int32_t>(macroSlot.getInt())),
      DenseIntElementsAttr::get(
          RankedTensorType::get({256}, builder.getI32Type()), words),
      exponentWords, isBF16,
      builder.getDictionaryAttr(planBinding)};
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
      Operation *command = nullptr;
      if (plan.bf16) {
        command = WriteBF16WeightsOp::create(
            builder, plan.location, plan.route, plan.macro, plan.words,
            plan.exponentWords)
                       .getOperation();
      } else {
        command = WriteInt8WeightsOp::create(
                      builder, plan.location, plan.route, plan.macro, plan.words)
                      .getOperation();
      }
      command->setAttr(cim::CIMDialect::getPlanBindingAttrName(),
                       plan.planBinding);
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
    if (Attribute planBinding =
            op->getAttr(cim::CIMDialect::getPlanBindingAttrName())) {
      control->setAttr(cim::CIMDialect::getPlanBindingAttrName(), planBinding);
      weight->setAttr(cim::CIMDialect::getPlanBindingAttrName(), planBinding);
    }
    rewriter.eraseOp(op);
    return success();
  }
};

class LowerWriteBF16Weights final
    : public OpConversionPattern<WriteBF16WeightsOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(WriteBF16WeightsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ControlBF16PacketOp control = ControlBF16PacketOp::create(
        rewriter, op.getLoc(), adaptor.getRoute(), adaptor.getMacro());
    WeightExponentPacketOp exponent = WeightExponentPacketOp::create(
        rewriter, op.getLoc(), adaptor.getRoute(), adaptor.getExponentWords());
    CIMBF16WeightPacketOp weight = CIMBF16WeightPacketOp::create(
        rewriter, op.getLoc(), adaptor.getRoute(), adaptor.getWords());
    if (Attribute planBinding =
            op->getAttr(cim::CIMDialect::getPlanBindingAttrName())) {
      control->setAttr(cim::CIMDialect::getPlanBindingAttrName(), planBinding);
      exponent->setAttr(cim::CIMDialect::getPlanBindingAttrName(), planBinding);
      weight->setAttr(cim::CIMDialect::getPlanBindingAttrName(), planBinding);
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
    target.addLegalOp<ControlInt8PacketOp, ControlBF16PacketOp, WorkOncePacketOp,
                      CIMInt8WeightPacketOp, WeightExponentPacketOp,
                      CIMBF16WeightPacketOp, WriteInputCacheInt8PacketOp,
                      WriteInputCacheBF16PacketOp,
                      ConfigureTestReturnRoutePacketOp,
                      ReadOutputCacheInt8PacketOp,
                      ReadOutputCacheBF16PacketOp>();
    target.markUnknownOpDynamicallyLegal([](Operation *op) {
      return op->getName().getDialectNamespace() !=
             CIMFrameDialect::getDialectNamespace();
    });

    RewritePatternSet patterns(&getContext());
    patterns.add<LowerStartInt8Once, LowerWriteInt8Weights,
                 LowerWriteBF16Weights>(&getContext());
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

class VerifyCIMFrame final : public impl::VerifyCIMFrameBase<VerifyCIMFrame> {
public:
  using Base::Base;

  void runOnOperation() override {
    if (failed(verifyCIMFrameModule(getOperation())))
      signalPassFailure();
  }
};
} // namespace
} // namespace mlir::cimframe
