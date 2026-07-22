//===- Passes.cpp - CIMFrame passes ----------------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIM22/Dialect/CIMFrame/Transforms/Passes.h"

#include "CIM22/Dialect/CIMFrame/IR/CIMFrameDialect.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::cimframe {
#define GEN_PASS_DEF_LOWERCIMFRAMECOMMANDSTOPACKETS
#define GEN_PASS_DEF_VERIFYCIMFRAME
#include "CIM22/Dialect/CIMFrame/Transforms/Passes.h.inc"

namespace {
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
    ControlInt8PacketOp::create(rewriter, op.getLoc(), adaptor.getRoute(),
                                adaptor.getMacro());
    CIMInt8WeightPacketOp::create(rewriter, op.getLoc(), adaptor.getRoute(),
                                  adaptor.getWords());
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
