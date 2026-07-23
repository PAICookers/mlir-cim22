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

#include "llvm/Support/MathExtras.h"

#include <algorithm>

namespace mlir::cim {
#define GEN_PASS_DEF_FORMCIMPROGRAM
#include "CIM22/Conversion/LinalgToCIM/Passes.h.inc"

namespace {
constexpr llvm::StringLiteral kMatMulIntegerMarker = "cim.onnx.matmul_integer";

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

bool hasCanonicalInt8ContractionBody(Region &region) {
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

bool hasCanonicalMatmulIndexingMaps(linalg::MatmulOp op) {
  SmallVector<AffineMap> maps = op.getIndexingMapsArray();
  SmallVector<AffineMap> expectedMaps =
      linalg::MatmulOp::getDefaultIndexingMaps(op.getContext());
  return maps == expectedMaps;
}

// TODO(CTQ-013): Linalg i21 arithmetic wraps on overflow, while CIM22 overflow
// behavior is not frozen. These software-only conversions assume every 64-term
// partial and final mathematical accumulation is exactly representable as
// signed i21.
bool isConvertible(linalg::MatvecOp op) {
  auto inputs = op.getDpsInputs();
  auto inits = op.getDpsInits();
  if (op->hasAttr(kMatMulIntegerMarker) || inputs.size() != 2 ||
      inits.size() != 1 || op->getNumResults() != 1)
    return false;

  auto weightType = dyn_cast<RankedTensorType>(inputs[0].getType());
  if (!weightType || weightType.getRank() != 2)
    return false;

  int64_t outputSize = weightType.getDimSize(0);
  if (outputSize <= 0)
    return false;

  int64_t reductionSize = weightType.getDimSize(1);
  if (reductionSize <= 0)
    return false;

  return hasType(inputs[0], {outputSize, reductionSize}, 8) &&
         hasType(inputs[1], {reductionSize}, 8) &&
         hasType(inits[0], {outputSize}, 21) &&
         hasType(op->getResult(0), {outputSize}, 21) && isZeroSplat(inits[0]) &&
         hasCanonicalInt8ContractionBody(op.getRegion()) &&
         hasCanonicalMatvecIndexingMaps(op);
}

bool isConvertible(linalg::MatmulOp op) {
  auto inputs = op.getDpsInputs();
  auto inits = op.getDpsInits();
  if (op->hasAttr(kMatMulIntegerMarker) || inputs.size() != 2 ||
      inits.size() != 1 || op->getNumResults() != 1)
    return false;

  auto weightType = dyn_cast<RankedTensorType>(inputs[0].getType());
  auto inputType = dyn_cast<RankedTensorType>(inputs[1].getType());
  if (!weightType || weightType.getRank() != 2 || !inputType ||
      inputType.getRank() != 2)
    return false;

  int64_t outputSize = weightType.getDimSize(0);
  int64_t reductionSize = weightType.getDimSize(1);
  int64_t columnCount = inputType.getDimSize(1);
  if (outputSize <= 0 || reductionSize <= 0 || columnCount <= 0)
    return false;

  return hasType(inputs[0], {outputSize, reductionSize}, 8) &&
         hasType(inputs[1], {reductionSize, columnCount}, 8) &&
         hasType(inits[0], {outputSize, columnCount}, 21) &&
         hasType(op->getResult(0), {outputSize, columnCount}, 21) &&
         isZeroSplat(inits[0]) && op.getCast() == linalg::TypeFn::cast_signed &&
         hasCanonicalInt8ContractionBody(op.getRegion()) &&
         hasCanonicalMatmulIndexingMaps(op);
}

constexpr int64_t kOutputTileSize = 16;
constexpr int64_t kReductionTileSize = 64;
constexpr int64_t kSignedI21Min = -1048576;
constexpr int64_t kSignedI21Max = 1048575;
constexpr int64_t kSignedI32Min = -2147483648;
constexpr int64_t kSignedI32Max = 2147483647;

enum class MatMulIntegerStatus {
  valid,
  invalid,
  partialRangeOverflow,
  resultRangeOverflow
};

bool hasMatMulIntegerMarker(Operation *op) {
  return op->hasAttr(kMatMulIntegerMarker);
}

bool hasCanonicalI32Int8ContractionBody(Region &region) {
  if (!region.hasOneBlock())
    return false;

  Block &block = region.front();
  if (block.getNumArguments() != 3 ||
      !block.getArgument(0).getType().isInteger(8) ||
      !block.getArgument(1).getType().isInteger(8) ||
      !block.getArgument(2).getType().isInteger(32) ||
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
         lhsExt.getOut().getType().isInteger(32) &&
         rhsExt.getIn() == block.getArgument(1) &&
         rhsExt.getOut().getType().isInteger(32) &&
         multiply.getLhs() == lhsExt.getOut() &&
         multiply.getRhs() == rhsExt.getOut() &&
         multiply.getResult().getType().isInteger(32) &&
         multiply.getOverflowFlags() == arith::IntegerOverflowFlags::none &&
         add.getLhs() == block.getArgument(2) &&
         add.getRhs() == multiply.getResult() &&
         add.getResult().getType().isInteger(32) &&
         add.getOverflowFlags() == arith::IntegerOverflowFlags::none &&
         yield.getValues().size() == 1 &&
         yield.getValues().front() == add.getResult();
}

MatMulIntegerStatus proveMatMulIntegerRanges(Value weight) {
  auto constant = weight.getDefiningOp<arith::ConstantOp>();
  auto elements = constant ? dyn_cast<DenseElementsAttr>(constant.getValue())
                           : DenseElementsAttr{};
  auto type = dyn_cast<RankedTensorType>(weight.getType());
  if (!elements || !type || type.getRank() != 2 ||
      elements.getNumElements() != type.getNumElements())
    return MatMulIntegerStatus::invalid;

  int64_t outputSize = type.getDimSize(0);
  int64_t reductionSize = type.getDimSize(1);

  SmallVector<int64_t> values;
  values.reserve(elements.getNumElements());
  for (APInt value : elements.getValues<APInt>())
    values.push_back(value.getSExtValue());

  for (int64_t output = 0; output < outputSize; ++output) {
    int64_t resultLower = 0;
    int64_t resultUpper = 0;
    for (int64_t reduction = 0; reduction < reductionSize;
         reduction += kReductionTileSize) {
      int64_t partialLower = 0;
      int64_t partialUpper = 0;
      int64_t tileSize =
          std::min(kReductionTileSize, reductionSize - reduction);
      for (int64_t offset = 0; offset < tileSize; ++offset) {
        int64_t weightValue =
            values[output * reductionSize + reduction + offset];
        partialUpper +=
            weightValue >= 0 ? 127 * weightValue : -128 * weightValue;
        partialLower +=
            weightValue >= 0 ? -128 * weightValue : 127 * weightValue;
      }
      if (partialLower < kSignedI21Min || partialUpper > kSignedI21Max)
        return MatMulIntegerStatus::partialRangeOverflow;
      resultLower += partialLower;
      resultUpper += partialUpper;
      if (resultLower < kSignedI32Min || resultUpper > kSignedI32Max)
        return MatMulIntegerStatus::resultRangeOverflow;
    }
  }
  return MatMulIntegerStatus::valid;
}

MatMulIntegerStatus getMatMulIntegerStatus(linalg::MatvecOp op) {
  auto inputs = op.getDpsInputs();
  auto inits = op.getDpsInits();
  auto weightType = inputs.size() == 2
                        ? dyn_cast<RankedTensorType>(inputs[0].getType())
                        : RankedTensorType{};
  if (!hasMatMulIntegerMarker(op) ||
      !isa<UnitAttr>(op->getAttr(kMatMulIntegerMarker)) || inputs.size() != 2 ||
      inits.size() != 1 || op->getNumResults() != 1 || !weightType ||
      weightType.getRank() != 2 || weightType.getDimSize(0) <= 0 ||
      weightType.getDimSize(1) <= 0)
    return MatMulIntegerStatus::invalid;
  int64_t outputSize = weightType.getDimSize(0);
  int64_t reductionSize = weightType.getDimSize(1);
  if (!hasType(inputs[0], {outputSize, reductionSize}, 8) ||
      !hasType(inputs[1], {reductionSize}, 8) ||
      !hasType(inits[0], {outputSize}, 32) ||
      !hasType(op->getResult(0), {outputSize}, 32) || !isZeroSplat(inits[0]) ||
      !hasCanonicalI32Int8ContractionBody(op.getRegion()) ||
      !hasCanonicalMatvecIndexingMaps(op))
    return MatMulIntegerStatus::invalid;
  return proveMatMulIntegerRanges(inputs[0]);
}

MatMulIntegerStatus getMatMulIntegerStatus(linalg::MatmulOp op) {
  auto inputs = op.getDpsInputs();
  auto inits = op.getDpsInits();
  auto weightType = inputs.size() == 2
                        ? dyn_cast<RankedTensorType>(inputs[0].getType())
                        : RankedTensorType{};
  auto inputType = inputs.size() == 2
                       ? dyn_cast<RankedTensorType>(inputs[1].getType())
                       : RankedTensorType{};
  if (!hasMatMulIntegerMarker(op) ||
      !isa<UnitAttr>(op->getAttr(kMatMulIntegerMarker)) || inputs.size() != 2 ||
      inits.size() != 1 || op->getNumResults() != 1 || !weightType ||
      weightType.getRank() != 2 || !inputType || inputType.getRank() != 2 ||
      weightType.getDimSize(0) <= 0 || weightType.getDimSize(1) <= 0 ||
      inputType.getDimSize(1) <= 0)
    return MatMulIntegerStatus::invalid;
  int64_t outputSize = weightType.getDimSize(0);
  int64_t reductionSize = weightType.getDimSize(1);
  int64_t columnCount = inputType.getDimSize(1);
  if (!hasType(inputs[0], {outputSize, reductionSize}, 8) ||
      !hasType(inputs[1], {reductionSize, columnCount}, 8) ||
      !hasType(inits[0], {outputSize, columnCount}, 32) ||
      !hasType(op->getResult(0), {outputSize, columnCount}, 32) ||
      !isZeroSplat(inits[0]) || op.getCast() != linalg::TypeFn::cast_signed ||
      !hasCanonicalI32Int8ContractionBody(op.getRegion()) ||
      !hasCanonicalMatmulIndexingMaps(op))
    return MatMulIntegerStatus::invalid;
  return proveMatMulIntegerRanges(inputs[0]);
}

LogicalResult rejectMatMulInteger(Operation *op, MatMulIntegerStatus status) {
  if (status == MatMulIntegerStatus::partialRangeOverflow)
    op->emitError("ONNX MatMulInteger violates CTQ-013: a constant K<=64 "
                  "partial is outside signed i21");
  else if (status == MatMulIntegerStatus::resultRangeOverflow)
    op->emitError("ONNX MatMulInteger constant-weight range proof exceeds "
                  "signed i32 result");
  else
    op->emitError("invalid ONNX MatMulInteger normalized i32 contract");
  return failure();
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
    int64_t reductionSize = weightType.getDimSize(1);
    if (reductionSize != 64) {
      Location location = op.getLoc();
      uint64_t tileCount =
          llvm::divideCeil(static_cast<uint64_t>(reductionSize), uint64_t{64});
      unsigned accumulationWidth = 21 + llvm::Log2_64_Ceil(tileCount);
      auto inputTileType =
          RankedTensorType::get({64}, weightType.getElementType());
      auto weightTileType =
          RankedTensorType::get({16, 64}, weightType.getElementType());
      auto resultType = cast<RankedTensorType>(op->getResult(0).getType());
      auto partialType =
          RankedTensorType::get({16}, resultType.getElementType());
      auto accumulationType = RankedTensorType::get(
          {16}, IntegerType::get(op.getContext(), accumulationWidth));
      SmallVector<Value> tileResults;
      tileResults.reserve(outputSize / 16 + (outputSize % 16 != 0));

      for (int64_t outputOffset = 0; outputOffset < outputSize;
           outputOffset += 16) {
        int64_t tileOutputSize =
            outputSize - outputOffset < 16 ? outputSize - outputOffset : 16;
        Value zero;
        Value sum;
        for (uint64_t reductionTile = 0; reductionTile < tileCount;
             ++reductionTile) {
          int64_t reductionOffset = static_cast<int64_t>(reductionTile * 64);
          int64_t tileReductionSize = reductionSize - reductionOffset < 64
                                          ? reductionSize - reductionOffset
                                          : 64;
          auto inputSliceType = RankedTensorType::get(
              {tileReductionSize}, weightType.getElementType());
          SmallVector<OpFoldResult> inputOffsets{
              rewriter.getIndexAttr(reductionOffset)};
          SmallVector<OpFoldResult> inputSizes{
              rewriter.getIndexAttr(tileReductionSize)};
          SmallVector<OpFoldResult> inputStrides{rewriter.getIndexAttr(1)};
          auto inputSlice = tensor::ExtractSliceOp::create(
              rewriter, location, inputSliceType, inputs[1], inputOffsets,
              inputSizes, inputStrides);

          auto weightSliceType = RankedTensorType::get(
              {tileOutputSize, tileReductionSize}, weightType.getElementType());
          SmallVector<OpFoldResult> weightOffsets{
              rewriter.getIndexAttr(outputOffset),
              rewriter.getIndexAttr(reductionOffset)};
          SmallVector<OpFoldResult> weightSizes{
              rewriter.getIndexAttr(tileOutputSize),
              rewriter.getIndexAttr(tileReductionSize)};
          SmallVector<OpFoldResult> weightStrides{rewriter.getIndexAttr(1),
                                                  rewriter.getIndexAttr(1)};
          auto weightSlice = tensor::ExtractSliceOp::create(
              rewriter, location, weightSliceType, inputs[0], weightOffsets,
              weightSizes, weightStrides);
          Value inputTile = inputSlice.getResult();
          Value weightTile = weightSlice.getResult();
          if (tileReductionSize < 64) {
            if (!zero)
              zero = arith::ConstantOp::create(
                  rewriter, location,
                  rewriter.getIntegerAttr(weightType.getElementType(), 0));
            SmallVector<OpFoldResult> low{rewriter.getIndexAttr(0)};
            SmallVector<OpFoldResult> high{
                rewriter.getIndexAttr(64 - tileReductionSize)};
            inputTile = tensor::PadOp::create(rewriter, location, inputTileType,
                                              inputTile, low, high, zero,
                                              /*nofold=*/false);
          }
          if (tileOutputSize < 16 || tileReductionSize < 64) {
            if (!zero)
              zero = arith::ConstantOp::create(
                  rewriter, location,
                  rewriter.getIntegerAttr(weightType.getElementType(), 0));
            SmallVector<OpFoldResult> low{rewriter.getIndexAttr(0),
                                          rewriter.getIndexAttr(0)};
            SmallVector<OpFoldResult> high{
                rewriter.getIndexAttr(16 - tileOutputSize),
                rewriter.getIndexAttr(64 - tileReductionSize)};
            weightTile = tensor::PadOp::create(
                rewriter, location, weightTileType, weightTile, low, high, zero,
                /*nofold=*/false);
          }

          Value partial = VMMOp::create(rewriter, location, partialType,
                                        inputTile, weightTile);
          if (tileCount == 1) {
            sum = partial;
            continue;
          }
          Value extended = arith::ExtSIOp::create(rewriter, location,
                                                  accumulationType, partial);
          sum = sum ? arith::AddIOp::create(rewriter, location, sum, extended)
                    : extended;
        }

        if (tileCount == 1)
          tileResults.push_back(sum);
        else
          tileResults.push_back(
              arith::TruncIOp::create(rewriter, location, partialType, sum));
      }

      Value tiledResult = tileResults.front();
      if (tileResults.size() > 1)
        tiledResult =
            tensor::ConcatOp::create(rewriter, location, 0, tileResults);

      if (outputSize % 16 == 0) {
        rewriter.replaceOp(op, tiledResult);
        return success();
      }

      SmallVector<OpFoldResult> offsets{rewriter.getIndexAttr(0)};
      SmallVector<OpFoldResult> sizes{rewriter.getIndexAttr(outputSize)};
      SmallVector<OpFoldResult> strides{rewriter.getIndexAttr(1)};
      rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
          op, resultType, tiledResult, offsets, sizes, strides);
      return success();
    }

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
    int64_t tailSize = outputSize % 16;
    int64_t fullSize = outputSize - tailSize;
    tileResults.reserve(fullSize / 16 + (tailSize != 0));
    for (int64_t offset = 0; offset < fullSize; offset += 16) {
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

    if (tailSize != 0) {
      auto tailType =
          RankedTensorType::get({tailSize, 64}, weightType.getElementType());
      SmallVector<OpFoldResult> offsets{rewriter.getIndexAttr(fullSize),
                                        rewriter.getIndexAttr(0)};
      SmallVector<OpFoldResult> sizes{rewriter.getIndexAttr(tailSize),
                                      rewriter.getIndexAttr(64)};
      SmallVector<OpFoldResult> strides{rewriter.getIndexAttr(1),
                                        rewriter.getIndexAttr(1)};
      auto tail = tensor::ExtractSliceOp::create(
          rewriter, location, tailType, inputs[0], offsets, sizes, strides);
      Value zero = arith::ConstantOp::create(
          rewriter, location,
          rewriter.getIntegerAttr(weightType.getElementType(), 0));
      SmallVector<OpFoldResult> low{rewriter.getIndexAttr(0),
                                    rewriter.getIndexAttr(0)};
      SmallVector<OpFoldResult> high{rewriter.getIndexAttr(16 - tailSize),
                                     rewriter.getIndexAttr(0)};
      auto padded = tensor::PadOp::create(rewriter, location, weightTileType,
                                          tail.getResult(), low, high, zero,
                                          /*nofold=*/false);
      tileResults.push_back(VMMOp::create(rewriter, location, resultTileType,
                                          inputs[1], padded.getResult()));
    }

    Value tiledResult = tileResults.front();
    if (tileResults.size() > 1)
      tiledResult =
          tensor::ConcatOp::create(rewriter, location, 0, tileResults);

    if (tailSize == 0) {
      rewriter.replaceOp(op, tiledResult);
      return success();
    }

    SmallVector<OpFoldResult> offsets{rewriter.getIndexAttr(0)};
    SmallVector<OpFoldResult> sizes{rewriter.getIndexAttr(outputSize)};
    SmallVector<OpFoldResult> strides{rewriter.getIndexAttr(1)};
    rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
        op, cast<RankedTensorType>(op->getResult(0).getType()), tiledResult,
        offsets, sizes, strides);
    return success();
  }
};

class FormMatmul final : public OpConversionPattern<linalg::MatmulOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::MatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isConvertible(op))
      return rewriter.notifyMatchFailure(op,
                                         "not an exact CIM MatMul candidate");

    Location location = op.getLoc();
    auto inputs = adaptor.getInputs();
    auto weightType = cast<RankedTensorType>(inputs[0].getType());
    auto inputType = cast<RankedTensorType>(inputs[1].getType());
    int64_t outputSize = weightType.getDimSize(0);
    int64_t reductionSize = weightType.getDimSize(1);
    int64_t columnCount = inputType.getDimSize(1);
    auto resultType = cast<RankedTensorType>(op->getResult(0).getType());
    auto columnType =
        RankedTensorType::get({reductionSize}, inputType.getElementType());
    auto columnResultType =
        RankedTensorType::get({outputSize}, resultType.getElementType());
    auto expandedResultType =
        RankedTensorType::get({outputSize, 1}, resultType.getElementType());
    auto zeroAttr = DenseElementsAttr::get(
        columnResultType,
        rewriter.getIntegerAttr(resultType.getElementType(), 0));
    Value zero = arith::ConstantOp::create(rewriter, location, columnResultType,
                                           zeroAttr);
    SmallVector<ReassociationIndices> reassociation{{0, 1}};
    SmallVector<Value> expandedColumns;
    expandedColumns.reserve(columnCount);

    for (int64_t column = 0; column < columnCount; ++column) {
      SmallVector<OpFoldResult> offsets{rewriter.getIndexAttr(0),
                                        rewriter.getIndexAttr(column)};
      SmallVector<OpFoldResult> sizes{rewriter.getIndexAttr(reductionSize),
                                      rewriter.getIndexAttr(1)};
      SmallVector<OpFoldResult> strides{rewriter.getIndexAttr(1),
                                        rewriter.getIndexAttr(1)};
      auto inputColumn = tensor::ExtractSliceOp::create(
          rewriter, location, columnType, inputs[1], offsets, sizes, strides);
      auto matvec = linalg::MatvecOp::create(
          rewriter, location, TypeRange{columnResultType},
          ValueRange{inputs[0], inputColumn.getResult()}, ValueRange{zero});
      expandedColumns.push_back(
          tensor::ExpandShapeOp::create(rewriter, location, expandedResultType,
                                        matvec.getResult(0), reassociation));
    }

    if (columnCount == 1) {
      rewriter.replaceOp(op, expandedColumns.front());
      return success();
    }

    rewriter.replaceOpWithNewOp<tensor::ConcatOp>(op, 1, expandedColumns);
    return success();
  }
};

class FormMatMulIntegerMatvec final
    : public OpConversionPattern<linalg::MatvecOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::MatvecOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!hasMatMulIntegerMarker(op))
      return rewriter.notifyMatchFailure(op,
                                         "not an ONNX MatMulInteger matvec");

    MatMulIntegerStatus status = getMatMulIntegerStatus(op);
    if (status != MatMulIntegerStatus::valid)
      return rejectMatMulInteger(op, status);

    Location location = op.getLoc();
    auto inputs = adaptor.getInputs();
    auto weightType = cast<RankedTensorType>(inputs[0].getType());
    auto opResultType = cast<RankedTensorType>(op->getResult(0).getType());
    int64_t outputSize = weightType.getDimSize(0);
    int64_t reductionSize = weightType.getDimSize(1);
    auto inputTileType = RankedTensorType::get(
        {kReductionTileSize}, IntegerType::get(op.getContext(), 8));
    auto weightTileType =
        RankedTensorType::get({kOutputTileSize, kReductionTileSize},
                              IntegerType::get(op.getContext(), 8));
    auto partialType = RankedTensorType::get(
        {kOutputTileSize}, IntegerType::get(op.getContext(), 21));
    auto resultTileType = RankedTensorType::get(
        {kOutputTileSize}, IntegerType::get(op.getContext(), 32));
    SmallVector<Value> outputTiles;
    outputTiles.reserve(
        llvm::divideCeil(static_cast<uint64_t>(outputSize),
                         static_cast<uint64_t>(kOutputTileSize)));
    Value zero = arith::ConstantOp::create(
        rewriter, location,
        rewriter.getIntegerAttr(weightType.getElementType(), 0));

    for (int64_t outputOffset = 0; outputOffset < outputSize;
         outputOffset += kOutputTileSize) {
      int64_t tileOutputSize =
          std::min(kOutputTileSize, outputSize - outputOffset);
      Value sum;
      for (int64_t reductionOffset = 0; reductionOffset < reductionSize;
           reductionOffset += kReductionTileSize) {
        int64_t tileReductionSize =
            std::min(kReductionTileSize, reductionSize - reductionOffset);
        auto inputSliceType = RankedTensorType::get(
            {tileReductionSize}, weightType.getElementType());
        SmallVector<OpFoldResult> inputOffsets{
            rewriter.getIndexAttr(reductionOffset)};
        SmallVector<OpFoldResult> inputSizes{
            rewriter.getIndexAttr(tileReductionSize)};
        SmallVector<OpFoldResult> inputStrides{rewriter.getIndexAttr(1)};
        Value inputTile = tensor::ExtractSliceOp::create(
            rewriter, location, inputSliceType, inputs[1], inputOffsets,
            inputSizes, inputStrides);

        auto weightSliceType = RankedTensorType::get(
            {tileOutputSize, tileReductionSize}, weightType.getElementType());
        SmallVector<OpFoldResult> weightOffsets{
            rewriter.getIndexAttr(outputOffset),
            rewriter.getIndexAttr(reductionOffset)};
        SmallVector<OpFoldResult> weightSizes{
            rewriter.getIndexAttr(tileOutputSize),
            rewriter.getIndexAttr(tileReductionSize)};
        SmallVector<OpFoldResult> weightStrides{rewriter.getIndexAttr(1),
                                                rewriter.getIndexAttr(1)};
        Value weightTile = tensor::ExtractSliceOp::create(
            rewriter, location, weightSliceType, inputs[0], weightOffsets,
            weightSizes, weightStrides);

        if (tileReductionSize < kReductionTileSize) {
          SmallVector<OpFoldResult> low{rewriter.getIndexAttr(0)};
          SmallVector<OpFoldResult> high{
              rewriter.getIndexAttr(kReductionTileSize - tileReductionSize)};
          inputTile = tensor::PadOp::create(rewriter, location, inputTileType,
                                            inputTile, low, high, zero,
                                            /*nofold=*/false);
        }
        if (tileOutputSize < kOutputTileSize ||
            tileReductionSize < kReductionTileSize) {
          SmallVector<OpFoldResult> low{rewriter.getIndexAttr(0),
                                        rewriter.getIndexAttr(0)};
          SmallVector<OpFoldResult> high{
              rewriter.getIndexAttr(kOutputTileSize - tileOutputSize),
              rewriter.getIndexAttr(kReductionTileSize - tileReductionSize)};
          weightTile = tensor::PadOp::create(rewriter, location, weightTileType,
                                             weightTile, low, high, zero,
                                             /*nofold=*/false);
        }

        Value partial = VMMOp::create(rewriter, location, partialType,
                                      inputTile, weightTile);
        Value extended =
            arith::ExtSIOp::create(rewriter, location, resultTileType, partial);
        sum = sum ? arith::AddIOp::create(rewriter, location, sum, extended)
                  : extended;
      }
      outputTiles.push_back(sum);
    }

    Value tiledResult = outputTiles.front();
    if (outputTiles.size() > 1)
      tiledResult =
          tensor::ConcatOp::create(rewriter, location, 0, outputTiles);
    if (outputSize % kOutputTileSize == 0) {
      rewriter.replaceOp(op, tiledResult);
      return success();
    }

    SmallVector<OpFoldResult> offsets{rewriter.getIndexAttr(0)};
    SmallVector<OpFoldResult> sizes{rewriter.getIndexAttr(outputSize)};
    SmallVector<OpFoldResult> strides{rewriter.getIndexAttr(1)};
    rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
        op, opResultType, tiledResult, offsets, sizes, strides);
    return success();
  }
};

class FormMatMulIntegerMatmul final
    : public OpConversionPattern<linalg::MatmulOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::MatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!hasMatMulIntegerMarker(op))
      return rewriter.notifyMatchFailure(op,
                                         "not an ONNX MatMulInteger matmul");

    MatMulIntegerStatus status = getMatMulIntegerStatus(op);
    if (status != MatMulIntegerStatus::valid)
      return rejectMatMulInteger(op, status);

    Location location = op.getLoc();
    auto inputs = adaptor.getInputs();
    auto weightType = cast<RankedTensorType>(inputs[0].getType());
    auto inputType = cast<RankedTensorType>(inputs[1].getType());
    auto resultType = cast<RankedTensorType>(op->getResult(0).getType());
    int64_t outputSize = weightType.getDimSize(0);
    int64_t reductionSize = weightType.getDimSize(1);
    int64_t columnCount = inputType.getDimSize(1);
    auto columnType =
        RankedTensorType::get({reductionSize}, inputType.getElementType());
    auto columnResultType =
        RankedTensorType::get({outputSize}, resultType.getElementType());
    auto expandedResultType =
        RankedTensorType::get({outputSize, 1}, resultType.getElementType());
    auto zeroAttr = DenseElementsAttr::get(
        columnResultType,
        rewriter.getIntegerAttr(resultType.getElementType(), 0));
    Value zero = arith::ConstantOp::create(rewriter, location, columnResultType,
                                           zeroAttr);
    SmallVector<ReassociationIndices> reassociation{{0, 1}};
    SmallVector<Value> expandedColumns;
    expandedColumns.reserve(columnCount);

    for (int64_t column = 0; column < columnCount; ++column) {
      SmallVector<OpFoldResult> offsets{rewriter.getIndexAttr(0),
                                        rewriter.getIndexAttr(column)};
      SmallVector<OpFoldResult> sizes{rewriter.getIndexAttr(reductionSize),
                                      rewriter.getIndexAttr(1)};
      SmallVector<OpFoldResult> strides{rewriter.getIndexAttr(1),
                                        rewriter.getIndexAttr(1)};
      auto inputColumn = tensor::ExtractSliceOp::create(
          rewriter, location, columnType, inputs[1], offsets, sizes, strides);
      auto matvec = linalg::MatvecOp::create(
          rewriter, location, TypeRange{columnResultType},
          ValueRange{inputs[0], inputColumn.getResult()}, ValueRange{zero});
      matvec->setAttr(kMatMulIntegerMarker, rewriter.getUnitAttr());
      expandedColumns.push_back(
          tensor::ExpandShapeOp::create(rewriter, location, expandedResultType,
                                        matvec.getResult(0), reassociation));
    }

    if (columnCount == 1) {
      rewriter.replaceOp(op, expandedColumns.front());
      return success();
    }
    rewriter.replaceOpWithNewOp<tensor::ConcatOp>(op, 1, expandedColumns);
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
    target.addDynamicallyLegalOp<linalg::MatvecOp>([](linalg::MatvecOp op) {
      return !isConvertible(op) && !hasMatMulIntegerMarker(op);
    });
    target.addDynamicallyLegalOp<linalg::MatmulOp>([](linalg::MatmulOp op) {
      return !isConvertible(op) && !hasMatMulIntegerMarker(op);
    });
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    RewritePatternSet patterns(&getContext());
    patterns.add<FormMatvec, FormMatmul, FormMatMulIntegerMatvec,
                 FormMatMulIntegerMatmul>(&getContext());
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace
} // namespace mlir::cim
