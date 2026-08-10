//===- Passes.cpp - Linalg to CIM conversion ------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIM22/Conversion/LinalgToCIM/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <limits>
#include <optional>

namespace mlir::cim {
#define GEN_PASS_DEF_FORMCIMPROGRAM
#define GEN_PASS_DEF_MATERIALIZECIMINVOCATION
#define GEN_PASS_DEF_MATERIALIZECIMSCHEDULE
#define GEN_PASS_DEF_NORMALIZECIMCONV
#include "CIM22/Conversion/LinalgToCIM/Passes.h.inc"

namespace {
constexpr llvm::StringLiteral kMatMulIntegerMarker = "cim.onnx.matmul_integer";
constexpr llvm::StringLiteral kConvIntegerMarker = "cim.onnx.conv_integer";
constexpr llvm::StringLiteral kMatTileMarker = "__cim_m_tile";
constexpr llvm::StringLiteral kTileAttrs[] = {"m_tile", "n_tile", "k_tile"};
constexpr llvm::StringLiteral kScheduleAttrs[] = {"work_id", "group_id"};

struct TileIdentity {
  int64_t m;
  int64_t n;
  int64_t k;
};

FailureOr<DenseElementsAttr> evaluateDenseTensor(Value value) {
  if (auto constant = value.getDefiningOp<arith::ConstantOp>())
    if (auto elements = dyn_cast<DenseElementsAttr>(constant.getValue()))
      return elements;

  if (auto collapse = value.getDefiningOp<tensor::CollapseShapeOp>()) {
    FailureOr<DenseElementsAttr> source =
        evaluateDenseTensor(collapse.getSrc());
    auto resultType = dyn_cast<RankedTensorType>(collapse.getType());
    if (failed(source) || !resultType ||
        (*source).getNumElements() != resultType.getNumElements())
      return failure();
    return (*source).reshape(resultType);
  }

  if (auto slice = value.getDefiningOp<tensor::ExtractSliceOp>()) {
    FailureOr<DenseElementsAttr> source =
        evaluateDenseTensor(slice.getSource());
    if (failed(source))
      return failure();
    auto resultType = dyn_cast<RankedTensorType>(slice.getType());
    auto sourceType = dyn_cast<RankedTensorType>((*source).getType());
    ArrayRef<int64_t> offsets = slice.getStaticOffsets();
    ArrayRef<int64_t> sizes = slice.getStaticSizes();
    ArrayRef<int64_t> strides = slice.getStaticStrides();
    if (!resultType || !sourceType || resultType.getRank() != 2 ||
        sourceType.getRank() != 2 || offsets.size() != 2 || sizes.size() != 2 ||
        strides.size() != 2 ||
        llvm::is_contained(offsets, ShapedType::kDynamic) ||
        llvm::is_contained(sizes, ShapedType::kDynamic) ||
        llvm::is_contained(strides, ShapedType::kDynamic))
      return failure();

    SmallVector<APInt> sourceValues((*source).getValues<APInt>());
    SmallVector<APInt> resultValues;
    resultValues.reserve(resultType.getNumElements());
    int64_t sourceColumns = sourceType.getDimSize(1);
    for (int64_t row = 0; row < sizes[0]; ++row)
      for (int64_t column = 0; column < sizes[1]; ++column) {
        int64_t sourceRow = offsets[0] + row * strides[0];
        int64_t sourceColumn = offsets[1] + column * strides[1];
        resultValues.push_back(
            sourceValues[sourceRow * sourceColumns + sourceColumn]);
      }
    return DenseElementsAttr::get(resultType, resultValues);
  }

  if (auto pad = value.getDefiningOp<tensor::PadOp>()) {
    FailureOr<DenseElementsAttr> source = evaluateDenseTensor(pad.getSource());
    if (failed(source))
      return failure();
    auto resultType = dyn_cast<RankedTensorType>(pad.getType());
    auto sourceType = dyn_cast<RankedTensorType>((*source).getType());
    ArrayRef<int64_t> low = pad.getStaticLow();
    ArrayRef<int64_t> high = pad.getStaticHigh();
    Value padding = pad.getConstantPaddingValue();
    auto constant = padding ? padding.getDefiningOp<arith::ConstantOp>()
                            : arith::ConstantOp{};
    auto scalar =
        constant ? dyn_cast<IntegerAttr>(constant.getValue()) : IntegerAttr{};
    if (!resultType || !sourceType || resultType.getRank() != 2 ||
        sourceType.getRank() != 2 || low.size() != 2 || high.size() != 2 ||
        llvm::is_contained(low, ShapedType::kDynamic) ||
        llvm::is_contained(high, ShapedType::kDynamic) || !scalar)
      return failure();

    SmallVector<APInt> sourceValues((*source).getValues<APInt>());
    SmallVector<APInt> resultValues(resultType.getNumElements(),
                                    scalar.getValue());
    int64_t sourceColumns = sourceType.getDimSize(1);
    int64_t resultColumns = resultType.getDimSize(1);
    for (int64_t row = 0; row < sourceType.getDimSize(0); ++row)
      for (int64_t column = 0; column < sourceColumns; ++column)
        resultValues[(row + low[0]) * resultColumns + column + low[1]] =
            sourceValues[row * sourceColumns + column];
    return DenseElementsAttr::get(resultType, resultValues);
  }
  return failure();
}

int64_t getMatTile(Operation *op) {
  if (auto attribute = op->getAttrOfType<IntegerAttr>(kMatTileMarker))
    return attribute.getInt();
  return 0;
}

VMMOp createIdentifiedVMM(OpBuilder &builder, Location location,
                          Type resultType, Value input, Value weight,
                          int64_t mTile, int64_t nTile, int64_t kTile) {
  auto vmm = VMMOp::create(builder, location, resultType, input, weight);
  vmm->setAttr("m_tile", builder.getI64IntegerAttr(mTile));
  vmm->setAttr("n_tile", builder.getI64IntegerAttr(nTile));
  vmm->setAttr("k_tile", builder.getI64IntegerAttr(kTile));
  return vmm;
}

unsigned countPresent(Operation *op,
                      ArrayRef<llvm::StringLiteral> attributeNames) {
  return llvm::count_if(attributeNames,
                        [op](StringRef name) { return op->hasAttr(name); });
}

std::optional<int64_t> readNonNegativeI64(VMMOp op, StringRef name) {
  Attribute attribute = op->getAttr(name);
  auto integer = dyn_cast_or_null<IntegerAttr>(attribute);
  if (!integer || !integer.getType().isSignlessInteger(64)) {
    op.emitOpError("expects '") << name << "' to be an i64 attribute";
    return std::nullopt;
  }
  if (integer.getInt() < 0) {
    op.emitOpError("expects '") << name << "' to be non-negative";
    return std::nullopt;
  }
  return integer.getInt();
}

std::optional<TileIdentity> readTileIdentity(VMMOp op) {
  if (countPresent(op, kTileAttrs) != std::size(kTileAttrs)) {
    op.emitOpError("materialize-cim-schedule requires complete m_tile, "
                   "n_tile, and k_tile identity");
    return std::nullopt;
  }
  auto m = readNonNegativeI64(op, "m_tile");
  auto n = readNonNegativeI64(op, "n_tile");
  auto k = readNonNegativeI64(op, "k_tile");
  if (!m || !n || !k)
    return std::nullopt;
  return TileIdentity{*m, *n, *k};
}

int64_t getGroupId(int64_t workId) {
  // TODO(CTQ-031): Keep software-only groups two-wide; M4 maps each group to
  // one core until cross-core waves are known.
  return workId / 2;
}

bool dependsOn(Operation *operation, Operation *possibleDependency) {
  SmallVector<Operation *> worklist;
  for (Value operand : operation->getOperands())
    if (Operation *definition = operand.getDefiningOp())
      worklist.push_back(definition);

  llvm::SmallPtrSet<Operation *, 16> visited;
  while (!worklist.empty()) {
    Operation *current = worklist.pop_back_val();
    if (current == possibleDependency)
      return true;
    if (!visited.insert(current).second)
      continue;
    for (Value operand : current->getOperands())
      if (Operation *definition = operand.getDefiningOp())
        worklist.push_back(definition);
  }
  return false;
}

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

bool isEvaluatedZeroSplat(Value value) {
  FailureOr<DenseElementsAttr> elements = evaluateDenseTensor(value);
  return succeeded(elements) && (*elements).isSplat() &&
         (*elements).getSplatValue<APInt>().isZero();
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
  FailureOr<DenseElementsAttr> elements = evaluateDenseTensor(weight);
  auto type = dyn_cast<RankedTensorType>(weight.getType());
  if (failed(elements) || !type || type.getRank() != 2 ||
      (*elements).getNumElements() != type.getNumElements())
    return MatMulIntegerStatus::invalid;

  int64_t outputSize = type.getDimSize(0);
  int64_t reductionSize = type.getDimSize(1);

  SmallVector<int64_t> values;
  values.reserve((*elements).getNumElements());
  for (APInt value : (*elements).getValues<APInt>())
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
      !hasType(op->getResult(0), {outputSize}, 32) ||
      !isEvaluatedZeroSplat(inits[0]) ||
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
      !isEvaluatedZeroSplat(inits[0]) ||
      op.getCast() != linalg::TypeFn::cast_signed ||
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
    int64_t matTile = getMatTile(op);
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

        Value partial = createIdentifiedVMM(
            rewriter, location, partialType, inputTile, weightTile, matTile,
            outputOffset / kOutputTileSize,
            reductionOffset / kReductionTileSize);
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
      matvec->setAttr(kMatTileMarker, rewriter.getI64IntegerAttr(column));
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

LogicalResult normalizeCIMConv(linalg::Conv2DNchwFchwOp op,
                               PatternRewriter &rewriter) {
  Location location = op.getLoc();
  auto reject = [&](StringRef message) {
    emitError(location) << "invalid marked ConvInteger: " << message;
    return failure();
  };
  if (!isa<UnitAttr>(op->getAttr(kConvIntegerMarker)))
    return reject("expects a unit cim.onnx.conv_integer marker");

  auto inputs = op.getDpsInputs();
  auto inits = op.getDpsInits();
  if (inputs.size() != 2 || inits.size() != 1 || op->getNumResults() != 1)
    return reject("has an invalid marked ConvInteger operand contract");
  auto inputType = dyn_cast<RankedTensorType>(inputs[0].getType());
  auto weightType = dyn_cast<RankedTensorType>(inputs[1].getType());
  auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!inputType || !weightType || !resultType || inputType.getRank() != 4 ||
      weightType.getRank() != 4 || resultType.getRank() != 4)
    return reject("requires ranked NCHW/FCHW tensors");

  int64_t channels = weightType.getDimSize(1);
  int64_t filters = weightType.getDimSize(0);
  int64_t height = resultType.getDimSize(2);
  int64_t width = resultType.getDimSize(3);
  if (channels <= 0 || filters <= 0 || height <= 0 || width <= 0 ||
      channels > std::numeric_limits<int64_t>::max() / 9 ||
      height > std::numeric_limits<int64_t>::max() - 2 ||
      width > std::numeric_limits<int64_t>::max() - 2 ||
      height > std::numeric_limits<int64_t>::max() / width ||
      !hasType(inputs[0], {1, channels, height + 2, width + 2}, 8) ||
      !hasType(inputs[1], {filters, channels, 3, 3}, 8) ||
      !hasType(inits[0], {1, filters, height, width}, 32) ||
      !hasType(op->getResult(0), {1, filters, height, width}, 32) ||
      !isZeroSplat(inits[0]) || failed(evaluateDenseTensor(inputs[1])) ||
      !llvm::all_of(op.getStrides().getValues<int64_t>(),
                    [](int64_t value) { return value == 1; }) ||
      !llvm::all_of(op.getDilations().getValues<int64_t>(),
                    [](int64_t value) { return value == 1; }))
    return reject("does not match the frozen integer Conv profile");

  FailureOr<std::pair<Operation *, Operation *>> rewritten =
      linalg::rewriteInIm2Col(rewriter, op);
  if (failed(rewritten))
    return reject("upstream im2col rewrite failed");
  auto im2col = dyn_cast<linalg::GenericOp>(rewritten->first);
  auto restore = dyn_cast<tensor::ExpandShapeOp>(rewritten->second);
  auto contraction = restore
                         ? restore.getSrc().getDefiningOp<linalg::GenericOp>()
                         : linalg::GenericOp{};
  if (!im2col || !restore || !contraction)
    return reject("upstream im2col rewrite returned an unexpected IR shape");

  auto contractionInputs = contraction.getDpsInputs();
  auto contractionInits = contraction.getDpsInits();
  if (contractionInputs.size() != 2 || contractionInits.size() != 1)
    return reject("upstream im2col contraction has unexpected operands");
  auto colType = dyn_cast<RankedTensorType>(contractionInputs[1].getType());
  auto batchedResultType =
      dyn_cast<RankedTensorType>(contraction->getResult(0).getType());
  if (!colType || !batchedResultType ||
      colType.getShape() !=
          ArrayRef<int64_t>{1, channels * 9, height * width} ||
      batchedResultType.getShape() !=
          ArrayRef<int64_t>{1, filters, height * width})
    return reject("upstream im2col contraction has unexpected batch shape");

  rewriter.setInsertionPoint(contraction);
  SmallVector<ReassociationIndices> collapseBatch{{0, 1}, {2}};
  auto col2DType = RankedTensorType::get({channels * 9, height * width},
                                         colType.getElementType());
  auto result2DType = RankedTensorType::get({filters, height * width},
                                            batchedResultType.getElementType());
  auto col2D = tensor::CollapseShapeOp::create(
      rewriter, location, col2DType, contractionInputs[1], collapseBatch);
  auto init2D = tensor::CollapseShapeOp::create(
      rewriter, location, result2DType, contractionInits[0], collapseBatch);
  auto matmul = linalg::MatmulOp::create(
      rewriter, location, TypeRange{result2DType},
      ValueRange{contractionInputs[0], col2D.getResult()},
      ValueRange{init2D.getResult()});
  matmul->setAttr(kMatMulIntegerMarker, rewriter.getUnitAttr());
  auto expanded =
      tensor::ExpandShapeOp::create(rewriter, location, batchedResultType,
                                    matmul.getResult(0), collapseBatch);
  rewriter.replaceOp(contraction, expanded.getResult());
  return success();
}

class NormalizeCIMConv final
    : public impl::NormalizeCIMConvBase<NormalizeCIMConv> {
public:
  using Base::Base;

  void runOnOperation() override {
    SmallVector<linalg::Conv2DNchwFchwOp> candidates;
    getOperation()->walk([&](linalg::Conv2DNchwFchwOp op) {
      if (op->hasAttr(kConvIntegerMarker))
        candidates.push_back(op);
    });
    PatternRewriter rewriter(&getContext());
    for (linalg::Conv2DNchwFchwOp op : candidates) {
      rewriter.setInsertionPoint(op);
      if (failed(normalizeCIMConv(op, rewriter))) {
        signalPassFailure();
        return;
      }
    }
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

class MaterializeCIMSchedule final
    : public impl::MaterializeCIMScheduleBase<MaterializeCIMSchedule> {
public:
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp function = getOperation();
    SmallVector<VMMOp> vmms;
    function.walk([&](VMMOp op) { vmms.push_back(op); });
    if (vmms.empty())
      return;

    if (!function.getBody().hasOneBlock()) {
      function.emitError(
          "materialize-cim-schedule requires one straight-line block");
      return signalPassFailure();
    }
    Block *body = &function.getBody().front();
    if (llvm::any_of(vmms,
                     [body](VMMOp op) { return op->getBlock() != body; })) {
      function.emitError(
          "materialize-cim-schedule cannot represent nested control flow");
      return signalPassFailure();
    }

    SmallVector<TileIdentity> identities;
    identities.reserve(vmms.size());
    int64_t maxM = 0;
    int64_t maxN = 0;
    int64_t maxK = 0;
    bool anyScheduled = false;
    bool allScheduled = true;
    for (VMMOp vmm : vmms) {
      auto identity = readTileIdentity(vmm);
      if (!identity)
        return signalPassFailure();
      identities.push_back(*identity);
      maxM = std::max(maxM, identity->m);
      maxN = std::max(maxN, identity->n);
      maxK = std::max(maxK, identity->k);

      unsigned scheduleCount = countPresent(vmm, kScheduleAttrs);
      if (scheduleCount != 0 && scheduleCount != std::size(kScheduleAttrs)) {
        vmm.emitOpError("materialize-cim-schedule requires complete work_id "
                        "and group_id attributes");
        return signalPassFailure();
      }
      anyScheduled |= scheduleCount != 0;
      allScheduled &= scheduleCount != 0;
    }
    if (anyScheduled != allScheduled) {
      function.emitError("materialize-cim-schedule rejects mixed scheduled "
                         "and unscheduled cim.vmm operations");
      return signalPassFailure();
    }

    if (maxM == std::numeric_limits<int64_t>::max() ||
        maxN == std::numeric_limits<int64_t>::max() ||
        maxK == std::numeric_limits<int64_t>::max()) {
      function.emitError("materialize-cim-schedule tile extent overflows i64");
      return signalPassFailure();
    }
    int64_t mTiles = maxM + 1;
    int64_t nTiles = maxN + 1;
    int64_t kTiles = maxK + 1;
    int64_t mnTiles = 0;
    int64_t expectedWorkCount = 0;
    if (llvm::MulOverflow(mTiles, nTiles, mnTiles) ||
        llvm::MulOverflow(mnTiles, kTiles, expectedWorkCount)) {
      function.emitError("materialize-cim-schedule tile rectangle overflows "
                         "i64");
      return signalPassFailure();
    }

    llvm::DenseSet<int64_t> seenWork;
    seenWork.reserve(vmms.size());
    for (auto [index, identity] : llvm::enumerate(identities)) {
      int64_t workId =
          ((identity.m * nTiles) + identity.n) * kTiles + identity.k;
      if (!seenWork.insert(workId).second) {
        vmms[index].emitOpError(
            "materialize-cim-schedule rejects duplicate tile identity");
        return signalPassFailure();
      }
    }
    if (expectedWorkCount != static_cast<int64_t>(vmms.size())) {
      function.emitError("materialize-cim-schedule requires a complete "
                         "rectangular tile identity space");
      return signalPassFailure();
    }

    for (auto [index, identity] : llvm::enumerate(identities)) {
      int64_t workId =
          ((identity.m * nTiles) + identity.n) * kTiles + identity.k;
      if (workId != static_cast<int64_t>(index)) {
        vmms[index].emitOpError("materialize-cim-schedule requires contiguous "
                                "M-major/N-major/K-minor tile order");
        return signalPassFailure();
      }
    }

    for (size_t index = 0; index + 1 < vmms.size(); index += 2) {
      if (dependsOn(vmms[index], vmms[index + 1]) ||
          dependsOn(vmms[index + 1], vmms[index])) {
        vmms[index + 1].emitOpError(
            "materialize-cim-schedule cannot pair SSA-dependent VMM work");
        return signalPassFailure();
      }
    }

    Builder builder(function.getContext());
    for (auto [index, vmm] : llvm::enumerate(vmms)) {
      int64_t workId = static_cast<int64_t>(index);
      int64_t groupId = getGroupId(workId);
      if (allScheduled) {
        struct ExpectedAttribute {
          StringRef name;
          int64_t value;
        };
        ExpectedAttribute expected[] = {{"work_id", workId},
                                        {"group_id", groupId}};
        for (const ExpectedAttribute &attribute : expected) {
          auto actual = readNonNegativeI64(vmm, attribute.name);
          if (!actual)
            return signalPassFailure();
          if (*actual != attribute.value) {
            vmm.emitOpError("materialize-cim-schedule expects '")
                << attribute.name << "' = " << attribute.value << ", but got "
                << *actual;
            return signalPassFailure();
          }
        }
        continue;
      }
      vmm->setAttr("work_id", builder.getI64IntegerAttr(workId));
      vmm->setAttr("group_id", builder.getI64IntegerAttr(groupId));
    }
  }
};

constexpr llvm::StringLiteral kInvocationProvenanceAttrs[] = {
    "m_tile",   "n_tile",    "k_tile",     "work_id",
    "group_id", "core_slot", "macro_slot", "cim.mapping"};

void copyInvocationProvenance(Operation *from, Operation *to) {
  for (StringRef name : kInvocationProvenanceAttrs)
    to->setAttr(name, from->getAttr(name));
}

bool isInvocationOp(Operation *op) {
  return isa<ConfigureInputOp, ConfigureWeightOp, DispatchOp, OnceOp,
             ReadbackOp, GroupBarrierOp>(op);
}

LogicalResult verifySameInvocationWork(Operation *expected, Operation *actual) {
  for (StringRef name : kInvocationProvenanceAttrs)
    if (expected->getAttr(name) != actual->getAttr(name))
      return actual->emitOpError("expects '")
             << name << "' to match its configure_input operation";
  return success();
}

LogicalResult verifyInvocationStructure(func::FuncOp function) {
  if (!function.getBody().hasOneBlock())
    return function.emitError(
        "materialize-cim-invocation requires one straight-line block");

  SmallVector<Operation *> operations;
  for (Operation &op : function.getBody().front())
    if (isInvocationOp(&op))
      operations.push_back(&op);

  size_t cursor = 0;
  int64_t expectedWork = 0;
  int64_t expectedGroup = 0;
  while (cursor < operations.size()) {
    SmallVector<ConfigureInputOp> inputs;
    while (cursor < operations.size() &&
           isa<ConfigureInputOp>(operations[cursor]) && inputs.size() < 2) {
      auto input = cast<ConfigureInputOp>(operations[cursor++]);
      if (cursor >= operations.size() ||
          !isa<ConfigureWeightOp>(operations[cursor]))
        return input.emitOpError(
            "must be followed by a per-Macro configure_weight operation");
      Operation *weight = operations[cursor++];
      if (failed(verifySameInvocationWork(input, weight)))
        return failure();
      auto work = input->getAttrOfType<IntegerAttr>("work_id");
      auto group = input->getAttrOfType<IntegerAttr>("group_id");
      auto macro = input->getAttrOfType<IntegerAttr>("macro_slot");
      if (!work || work.getInt() != expectedWork || !group ||
          group.getInt() != expectedGroup || !macro ||
          macro.getInt() != static_cast<int64_t>(inputs.size()))
        return input.emitOpError(
            "does not match contiguous work/group/Macro ordering");
      inputs.push_back(input);
      ++expectedWork;
    }
    if (inputs.empty())
      return operations[cursor]->emitOpError(
          "expected group to start with configure_input/configure_weight");

    for (ConfigureInputOp input : inputs) {
      if (cursor >= operations.size() || !isa<DispatchOp>(operations[cursor]))
        return input.emitOpError(
            "expects one dispatch for each configured Macro payload");
      if (failed(verifySameInvocationWork(input, operations[cursor++])))
        return failure();
    }
    if (cursor >= operations.size() || !isa<OnceOp>(operations[cursor]))
      return inputs.front().emitOpError(
          "expects exactly one core-level once after all dispatches");
    Operation *once = operations[cursor++];
    if (once->getAttr("group_id") != inputs.front()->getAttr("group_id") ||
        once->getAttr("core_slot") != inputs.front()->getAttr("core_slot") ||
        once->getAttr("cim.mapping") != inputs.front()->getAttr("cim.mapping"))
      return once->emitOpError("does not match its invocation group");

    for (ConfigureInputOp input : inputs) {
      if (cursor >= operations.size() || !isa<ReadbackOp>(operations[cursor]))
        return input.emitOpError(
            "expects one readback for each configured Macro payload");
      if (failed(verifySameInvocationWork(input, operations[cursor++])))
        return failure();
    }
    if (cursor >= operations.size() || !isa<GroupBarrierOp>(operations[cursor]))
      return inputs.front().emitOpError("expects a terminating group barrier");
    Operation *barrier = operations[cursor++];
    if (barrier->getAttr("group_id") != inputs.front()->getAttr("group_id"))
      return barrier->emitOpError("does not match its invocation group");
    if (inputs.size() == 1 && cursor != operations.size())
      return inputs.front().emitOpError(
          "single-Macro group is only valid as the final group");
    ++expectedGroup;
  }
  return success();
}

class MaterializeCIMInvocation final
    : public impl::MaterializeCIMInvocationBase<MaterializeCIMInvocation> {
public:
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    struct WorkPlan {
      VMMOp vmm;
      DenseIntElementsAttr weight;
      int64_t workId;
      int64_t groupId;
      int64_t coreSlot;
      int64_t macroSlot;
    };
    struct FunctionPlan {
      func::FuncOp function;
      SmallVector<WorkPlan> works;
    };
    SmallVector<FunctionPlan> plans;

    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
      SmallVector<VMMOp> vmms;
      function.walk([&](VMMOp op) { vmms.push_back(op); });
      bool hasInvocation = false;
      function.walk(
          [&](Operation *op) { hasInvocation |= isInvocationOp(op); });
      if (vmms.empty()) {
        if (hasInvocation) {
          auto schema = function->getAttrOfType<IntegerAttr>(
              "cim.artifact_schema_version");
          if (!schema || !schema.getType().isSignlessInteger(64) ||
              schema.getInt() != 1) {
            function.emitError("materialize-cim-invocation expects "
                               "cim.artifact_schema_version = 1 : i64");
            return signalPassFailure();
          }
          if (failed(verifyInvocationStructure(function)))
            return signalPassFailure();
        }
        continue;
      }
      if (hasInvocation) {
        function.emitError(
            "materialize-cim-invocation rejects mixed VMM and invocation ops");
        return signalPassFailure();
      }
      if (!function.getBody().hasOneBlock()) {
        function.emitError(
            "materialize-cim-invocation requires one straight-line block");
        return signalPassFailure();
      }
      if (function->hasAttr("cim.artifact_schema_version")) {
        function.emitError("materialize-cim-invocation rejects stale artifact "
                           "schema attribute");
        return signalPassFailure();
      }

      FunctionPlan plan{function, {}};
      plan.works.reserve(vmms.size());
      for (auto [index, vmm] : llvm::enumerate(vmms)) {
        if (countPresent(vmm, kInvocationProvenanceAttrs) !=
            std::size(kInvocationProvenanceAttrs)) {
          vmm.emitOpError("materialize-cim-invocation requires complete mapped "
                          "work provenance");
          return signalPassFailure();
        }
        auto workId = readNonNegativeI64(vmm, "work_id");
        auto groupId = readNonNegativeI64(vmm, "group_id");
        auto coreSlot = readNonNegativeI64(vmm, "core_slot");
        auto macroSlot = readNonNegativeI64(vmm, "macro_slot");
        if (!workId || !groupId || !coreSlot || !macroSlot)
          return signalPassFailure();
        if (*workId != static_cast<int64_t>(index) || *groupId != *workId / 2 ||
            *macroSlot < 0 || *macroSlot > 1) {
          vmm.emitOpError(
              "materialize-cim-invocation rejects invalid work placement");
          return signalPassFailure();
        }
        FailureOr<DenseElementsAttr> weight =
            evaluateDenseTensor(vmm.getWeight());
        if (failed(weight) ||
            (*weight).getType() != vmm.getWeight().getType()) {
          vmm.emitOpError("materialize-cim-invocation requires a compile-time "
                          "constant weight tile");
          return signalPassFailure();
        }
        auto intWeight = dyn_cast<DenseIntElementsAttr>(*weight);
        if (!intWeight) {
          vmm.emitOpError("materialize-cim-invocation requires INT8 weight "
                          "elements");
          return signalPassFailure();
        }
        plan.works.push_back(
            {vmm, intWeight, *workId, *groupId, *coreSlot, *macroSlot});
      }

      for (size_t index = 0; index < plan.works.size(); index += 2) {
        WorkPlan &first = plan.works[index];
        if (index + 1 == plan.works.size())
          continue;
        WorkPlan &second = plan.works[index + 1];
        if (first.groupId != second.groupId ||
            first.coreSlot != second.coreSlot || first.macroSlot != 0 ||
            second.macroSlot != 1 ||
            first.vmm->getAttr("cim.mapping") !=
                second.vmm->getAttr("cim.mapping")) {
          second.vmm.emitOpError(
              "materialize-cim-invocation requires a same-route Macro 0/1 "
              "pair per two-work group");
          return signalPassFailure();
        }
      }
      plans.push_back(std::move(plan));
    }

    OpBuilder moduleBuilder(module.getContext());
    moduleBuilder.setInsertionPointToStart(module.getBody());
    for (FunctionPlan &plan : plans) {
      SmallVector<FlatSymbolRefAttr> resources;
      resources.reserve(plan.works.size());
      for (WorkPlan &work : plan.works) {
        std::string name =
            (Twine("__cim_weight_") + plan.function.getSymName() + "_w" +
             Twine(work.workId))
                .str();
        StaticWeightOp::create(moduleBuilder, work.vmm.getLoc(), name,
                               work.weight);
        resources.push_back(FlatSymbolRefAttr::get(module.getContext(), name));
      }

      plan.function->setAttr("cim.artifact_schema_version",
                             moduleBuilder.getI64IntegerAttr(1));
      for (size_t index = 0; index < plan.works.size(); index += 2) {
        size_t count = std::min<size_t>(2, plan.works.size() - index);
        WorkPlan &last = plan.works[index + count - 1];
        SmallVector<Operation *> earlyUsers;
        if (count == 2) {
          for (Operation *op = plan.works[index].vmm->getNextNode();
               op && op != last.vmm.getOperation(); op = op->getNextNode())
            if (dependsOn(op, plan.works[index].vmm))
              earlyUsers.push_back(op);
        }

        OpBuilder builder(last.vmm);
        SmallVector<ReadbackOp> readbacks;
        readbacks.reserve(count);
        for (size_t offset = 0; offset < count; ++offset) {
          WorkPlan &work = plan.works[index + offset];
          auto input = ConfigureInputOp::create(builder, work.vmm.getLoc(),
                                                work.vmm.getInput());
          copyInvocationProvenance(work.vmm, input);
          auto weight = ConfigureWeightOp::create(builder, work.vmm.getLoc(),
                                                  resources[index + offset]);
          copyInvocationProvenance(work.vmm, weight);
        }
        for (size_t offset = 0; offset < count; ++offset) {
          WorkPlan &work = plan.works[index + offset];
          auto dispatch = DispatchOp::create(builder, work.vmm.getLoc());
          copyInvocationProvenance(work.vmm, dispatch);
        }
        WorkPlan &first = plan.works[index];
        auto once = OnceOp::create(builder, first.vmm.getLoc());
        once->setAttr("group_id", first.vmm->getAttr("group_id"));
        once->setAttr("core_slot", first.vmm->getAttr("core_slot"));
        once->setAttr("cim.mapping", first.vmm->getAttr("cim.mapping"));
        for (size_t offset = 0; offset < count; ++offset) {
          WorkPlan &work = plan.works[index + offset];
          auto readback = ReadbackOp::create(builder, work.vmm.getLoc(),
                                             work.vmm.getResult().getType());
          copyInvocationProvenance(work.vmm, readback);
          readbacks.push_back(readback);
        }
        auto barrier = GroupBarrierOp::create(builder, first.vmm.getLoc());
        barrier->setAttr("group_id", first.vmm->getAttr("group_id"));

        Operation *anchor = barrier;
        for (Operation *op : earlyUsers) {
          op->moveAfter(anchor);
          anchor = op;
        }
        for (size_t offset = 0; offset < count; ++offset) {
          WorkPlan &work = plan.works[index + offset];
          work.vmm.getResult().replaceAllUsesWith(
              readbacks[offset].getResult());
        }
        for (size_t offset = 0; offset < count; ++offset)
          plan.works[index + offset].vmm.erase();
      }
      if (failed(verifyInvocationStructure(plan.function)))
        return signalPassFailure();
    }
  }
};
} // namespace
} // namespace mlir::cim
