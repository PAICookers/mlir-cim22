//===- Passes.cpp - Linalg to CIM conversion ------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIM22/Conversion/LinalgToCIM/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::cim {
#define GEN_PASS_DEF_FORMCIMPROGRAM
#include "CIM22/Conversion/LinalgToCIM/Passes.h.inc"

namespace {
bool hasType(Value value, ArrayRef<int64_t> shape, unsigned bitWidth) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  auto elementType =
      type ? dyn_cast<IntegerType>(type.getElementType()) : IntegerType{};
  return type && type.getShape() == shape && elementType &&
         elementType.getWidth() == bitWidth && elementType.isSignless();
}

bool isZeroSplat(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  auto elements = constant ? dyn_cast<DenseElementsAttr>(constant.getValue())
                           : DenseElementsAttr{};
  return elements && elements.isSplat() &&
         elements.getSplatValue<APInt>().isZero();
}

bool hasCanonicalInt8MatvecBody(linalg::MatvecOp op) {
  Region &region = op->getRegion(0);
  if (!region.hasOneBlock())
    return false;

  Block &block = region.front();
  if (block.getNumArguments() != 3 ||
      !block.getArgument(0).getType().isInteger(8) ||
      !block.getArgument(1).getType().isInteger(8) ||
      !block.getArgument(2).getType().isInteger(21) ||
      block.getOperations().size() != 5)
    return false;

  auto operation = block.begin();
  auto lhsExt = dyn_cast<arith::ExtSIOp>(&*operation++);
  auto rhsExt = dyn_cast<arith::ExtSIOp>(&*operation++);
  auto multiply = dyn_cast<arith::MulIOp>(&*operation++);
  auto add = dyn_cast<arith::AddIOp>(&*operation++);
  auto yield = dyn_cast<linalg::YieldOp>(&*operation);
  if (!lhsExt || !rhsExt || !multiply || !add || !yield)
    return false;

  return lhsExt.getIn() == block.getArgument(0) &&
         lhsExt.getOut().getType().isInteger(21) &&
         rhsExt.getIn() == block.getArgument(1) &&
         rhsExt.getOut().getType().isInteger(21) &&
         multiply.getLhs() == lhsExt.getOut() &&
         multiply.getRhs() == rhsExt.getOut() &&
         multiply.getResult().getType().isInteger(21) &&
         multiply.getOverflowFlags() == arith::IntegerOverflowFlags::none &&
         add.getLhs() == block.getArgument(2) &&
         add.getRhs() == multiply.getResult() &&
         add.getResult().getType().isInteger(21) &&
         add.getOverflowFlags() == arith::IntegerOverflowFlags::none &&
         yield.getValues().size() == 1 &&
         yield.getValues().front() == add.getResult();
}

bool hasCanonicalMatvecIndexingMaps(linalg::MatvecOp op) {
  SmallVector<AffineMap> maps = op.getIndexingMapsArray();
  MLIRContext *context = op.getContext();
  return maps.size() == 3 &&
         maps[0] == AffineMap::getMultiDimIdentityMap(2, context) &&
         maps[1] == AffineMap::get(2, 0, getAffineDimExpr(1, context)) &&
         maps[2] == AffineMap::get(2, 0, getAffineDimExpr(0, context));
}

// FIXME(CTQ-013): Linalg i21 arithmetic wraps on overflow, while CIM22
// overflow behavior is not frozen. This software-only conversion assumes every
// accumulation is exactly representable as signed i21.
bool isConvertible(linalg::MatvecOp op) {
  auto inputs = op.getDpsInputs();
  auto inits = op.getDpsInits();
  if (inputs.size() != 2 || inits.size() != 1 || op->getNumResults() != 1)
    return false;

  auto weightType = dyn_cast<RankedTensorType>(inputs[0].getType());
  if (!weightType || weightType.getRank() != 2)
    return false;

  int64_t outputSize = weightType.getDimSize(0);
  if (outputSize <= 0 || outputSize % 16 != 0)
    return false;

  return hasType(inputs[0], {outputSize, 64}, 8) &&
         hasType(inputs[1], {64}, 8) && hasType(inits[0], {outputSize}, 21) &&
         hasType(op->getResult(0), {outputSize}, 21) && isZeroSplat(inits[0]) &&
         hasCanonicalInt8MatvecBody(op) && hasCanonicalMatvecIndexingMaps(op);
}

class FormMatvec final : public OpConversionPattern<linalg::MatvecOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::MatvecOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isConvertible(op))
      return rewriter.notifyMatchFailure(op, "not an exact CIM VMM candidate");

    auto inputs = adaptor.getInputs();
    auto weightType = cast<RankedTensorType>(inputs[0].getType());
    int64_t outputSize = weightType.getDimSize(0);
    if (outputSize == 16) {
      rewriter.replaceOpWithNewOp<VMMOp>(op, op->getResult(0).getType(),
                                         inputs[1], inputs[0]);
      return success();
    }

    Location location = op.getLoc();
    auto weightTileType =
        RankedTensorType::get({16, 64}, weightType.getElementType());
    auto resultTileType = RankedTensorType::get(
        {16}, cast<ShapedType>(op->getResult(0).getType()).getElementType());
    SmallVector<Value> tileResults;
    tileResults.reserve(outputSize / 16);
    for (int64_t offset = 0; offset < outputSize; offset += 16) {
      SmallVector<OpFoldResult> offsets{rewriter.getIndexAttr(offset),
                                        rewriter.getIndexAttr(0)};
      SmallVector<OpFoldResult> sizes{rewriter.getIndexAttr(16),
                                      rewriter.getIndexAttr(64)};
      SmallVector<OpFoldResult> strides{rewriter.getIndexAttr(1),
                                        rewriter.getIndexAttr(1)};
      auto weightTile =
          tensor::ExtractSliceOp::create(rewriter, location, weightTileType,
                                         inputs[0], offsets, sizes, strides);
      tileResults.push_back(VMMOp::create(rewriter, location, resultTileType,
                                          inputs[1], weightTile.getResult()));
    }

    rewriter.replaceOpWithNewOp<tensor::ConcatOp>(op, 0, tileResults);
    return success();
  }
};

class FormCIMProgram final : public impl::FormCIMProgramBase<FormCIMProgram> {
public:
  using Base::Base;

  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addLegalDialect<CIMDialect>();
    target.addLegalDialect<tensor::TensorDialect>();
    target.addDynamicallyLegalOp<linalg::MatvecOp>(
        [](linalg::MatvecOp op) { return !isConvertible(op); });
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    RewritePatternSet patterns(&getContext());
    patterns.add<FormMatvec>(&getContext());
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace
} // namespace mlir::cim
