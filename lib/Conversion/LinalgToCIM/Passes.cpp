//===- Passes.cpp - Linalg to CIM conversion ------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIM22/Conversion/LinalgToCIM/Passes.h"
#include "CIM22/Conversion/LinalgToCIM/ExecutionPlanVerifier.h"
#include "CIM22/Dialect/CIM/IR/CIMDialect.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <optional>

namespace mlir::cim {
#define GEN_PASS_DEF_FOLDCIMINT8BIAS
#define GEN_PASS_DEF_FORMCIMPROGRAM
#define GEN_PASS_DEF_MATERIALIZECIMEXECUTIONPLAN
#define GEN_PASS_DEF_MATERIALIZECIMSCHEDULE
#define GEN_PASS_DEF_NORMALIZECIMCONV
#define GEN_PASS_DEF_PARTITIONCIMPROGRAM
#define GEN_PASS_DEF_VERIFYCIMEXECUTIONPLAN
#include "CIM22/Conversion/LinalgToCIM/Passes.h.inc"

namespace {
const llvm::StringRef kTileAttrs[] = {"m_tile", "n_tile", "k_tile"};
const llvm::StringRef kScheduleAttrs[] = {"work_id", "group_id"};

struct TileIdentity {
  int64_t m;
  int64_t n;
  int64_t k;
};

bool isExecutionPlanOp(Operation *op);

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
  if (auto attribute =
          op->getAttrOfType<IntegerAttr>(CIMDialect::getMatTileAttrName()))
    return attribute.getInt();
  return 0;
}

VMMOp createIdentifiedVMM(OpBuilder &builder, Location location,
                          Type resultType, Value input, Value weight,
                          int64_t segmentId, int64_t mTile, int64_t nTile,
                          int64_t kTile) {
  auto vmm = VMMOp::create(builder, location, resultType, input, weight);
  vmm->setAttr(CIMDialect::getSegmentIdAttrName(),
               builder.getI64IntegerAttr(segmentId));
  vmm->setAttr("m_tile", builder.getI64IntegerAttr(mTile));
  vmm->setAttr("n_tile", builder.getI64IntegerAttr(nTile));
  vmm->setAttr("k_tile", builder.getI64IntegerAttr(kTile));
  return vmm;
}

unsigned countPresent(Operation *op, ArrayRef<llvm::StringRef> attributeNames) {
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

bool dependsOn(Operation *operation, Operation *possibleDependency) {
  llvm::SetVector<Operation *> dependencies;
  return succeeded(getBackwardSlice(operation, &dependencies)) &&
         dependencies.contains(possibleDependency);
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

bool hasCanonicalInt8ContractionBody(Region &region,
                                     unsigned accumulatorWidth) {
  if (!region.hasOneBlock())
    return false;

  Block &block = region.front();
  if (block.getNumArguments() != 3 ||
      !block.getArgument(0).getType().isInteger(8) ||
      !block.getArgument(1).getType().isInteger(8) ||
      !block.getArgument(2).getType().isInteger(accumulatorWidth) ||
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
         lhsExt.getOut().getType().isInteger(accumulatorWidth) &&
         rhsExt.getIn() == block.getArgument(1) &&
         rhsExt.getOut().getType().isInteger(accumulatorWidth) &&
         multiply.getLhs() == lhsExt.getOut() &&
         multiply.getRhs() == rhsExt.getOut() &&
         multiply.getResult().getType().isInteger(accumulatorWidth) &&
         multiply.getOverflowFlags() == arith::IntegerOverflowFlags::none &&
         add.getLhs() == block.getArgument(2) &&
         add.getRhs() == multiply.getResult() &&
         add.getResult().getType().isInteger(accumulatorWidth) &&
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

// Linalg i21 arithmetic wraps on overflow, while CIM22 overflow behavior is
// not part of the current execution evidence. These software-only conversions
// assume every 64-term partial and final mathematical accumulation is exactly
// representable as signed i21.
bool isConvertible(linalg::MatvecOp op) {
  auto inputs = op.getDpsInputs();
  auto inits = op.getDpsInits();
  if (op->hasAttr(CIMDialect::getMatMulIntegerAttrName()) ||
      inputs.size() != 2 || inits.size() != 1 || op->getNumResults() != 1)
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
         hasCanonicalInt8ContractionBody(op.getRegion(), 21) &&
         hasCanonicalMatvecIndexingMaps(op);
}

bool isConvertible(linalg::MatmulOp op) {
  auto inputs = op.getDpsInputs();
  auto inits = op.getDpsInits();
  if (op->hasAttr(CIMDialect::getMatMulIntegerAttrName()) ||
      inputs.size() != 2 || inits.size() != 1 || op->getNumResults() != 1)
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
         hasCanonicalInt8ContractionBody(op.getRegion(), 21) &&
         hasCanonicalMatmulIndexingMaps(op);
}

constexpr int64_t kOutputTileSize = 16;
constexpr int64_t kReductionTileSize = 64;

enum class MatMulIntegerStatus { valid, invalid, partialRangeOverflow };

using IntegerRange = std::pair<int64_t, int64_t>;

bool hasMatMulIntegerMarker(Operation *op) {
  return op->hasAttr(CIMDialect::getMatMulIntegerAttrName());
}

SmallVector<IntegerRange> getMatMulIntegerInputRanges(Value input,
                                                      int64_t reductionSize) {
  SmallVector<IntegerRange> ranges(reductionSize, {-128, 127});
  Value matrix = input;
  if (auto slice = input.getDefiningOp<tensor::ExtractSliceOp>()) {
    auto resultType = dyn_cast<RankedTensorType>(slice.getType());
    auto sourceType = dyn_cast<RankedTensorType>(slice.getSource().getType());
    ArrayRef<int64_t> offsets = slice.getStaticOffsets();
    ArrayRef<int64_t> sizes = slice.getStaticSizes();
    ArrayRef<int64_t> strides = slice.getStaticStrides();
    if (!resultType || resultType.getRank() != 1 || !sourceType ||
        sourceType.getRank() != 2 || offsets.size() != 2 || sizes.size() != 2 ||
        strides.size() != 2 || offsets[0] != 0 || offsets[1] < 0 ||
        sizes[0] != reductionSize || sizes[1] != 1 || strides[0] != 1 ||
        strides[1] != 1)
      return ranges;
    matrix = slice.getSource();
  }

  auto concat = matrix.getDefiningOp<tensor::ConcatOp>();
  auto concatType = dyn_cast<RankedTensorType>(matrix.getType());
  if (!concat || concat.getDim() != 0 || !concatType ||
      concatType.getRank() != 2 || concatType.getDimSize(0) != reductionSize)
    return ranges;

  int64_t row = 0;
  for (Value operand : concat.getInputs()) {
    auto type = dyn_cast<RankedTensorType>(operand.getType());
    if (!type || type.getRank() != 2 || type.getDimSize(0) < 0 ||
        type.getDimSize(1) != concatType.getDimSize(1) ||
        row + type.getDimSize(0) > reductionSize)
      return SmallVector<IntegerRange>(reductionSize, {-128, 127});
    row += type.getDimSize(0);
  }
  if (row != reductionSize)
    return ranges;

  row = 0;
  for (Value operand : concat.getInputs()) {
    auto type = cast<RankedTensorType>(operand.getType());
    auto constant = operand.getDefiningOp<arith::ConstantOp>();
    auto elements = constant ? dyn_cast<DenseElementsAttr>(constant.getValue())
                             : DenseElementsAttr{};
    if (type.getDimSize(0) == 1 && elements && elements.isSplat() &&
        elements.getSplatValue<APInt>().isOne())
      ranges[row] = {1, 1};
    row += type.getDimSize(0);
  }
  return ranges;
}

MatMulIntegerStatus
proveMatMulIntegerRanges(DenseElementsAttr elements,
                         ArrayRef<IntegerRange> inputRanges) {
  auto type = dyn_cast<RankedTensorType>(elements.getType());
  if (!type || type.getRank() != 2 ||
      elements.getNumElements() != type.getNumElements() ||
      static_cast<int64_t>(inputRanges.size()) != type.getDimSize(1))
    return MatMulIntegerStatus::invalid;

  int64_t outputSize = type.getDimSize(0);
  int64_t reductionSize = type.getDimSize(1);
  SmallVector<int64_t> values;
  values.reserve(elements.getNumElements());
  for (APInt value : elements.getValues<APInt>())
    values.push_back(value.getSExtValue());

  for (int64_t output = 0; output < outputSize; ++output) {
    for (int64_t reduction = 0; reduction < reductionSize;
         reduction += kReductionTileSize) {
      int64_t partialLower = 0;
      int64_t partialUpper = 0;
      int64_t tileSize =
          std::min(kReductionTileSize, reductionSize - reduction);
      for (int64_t offset = 0; offset < tileSize; ++offset) {
        int64_t lane = reduction + offset;
        int64_t weightValue = values[output * reductionSize + lane];
        auto [inputLower, inputUpper] = inputRanges[lane];
        partialUpper += weightValue >= 0 ? inputUpper * weightValue
                                         : inputLower * weightValue;
        partialLower += weightValue >= 0 ? inputLower * weightValue
                                         : inputUpper * weightValue;
      }
      if (!llvm::isInt<21>(partialLower) || !llvm::isInt<21>(partialUpper))
        return MatMulIntegerStatus::partialRangeOverflow;
    }
  }
  return MatMulIntegerStatus::valid;
}

MatMulIntegerStatus proveMatMulIntegerRanges(Value weight, Value input) {
  FailureOr<DenseElementsAttr> elements = evaluateDenseTensor(weight);
  auto type = dyn_cast<RankedTensorType>(weight.getType());
  if (failed(elements) || !type || type.getRank() != 2 ||
      (*elements).getNumElements() != type.getNumElements())
    return MatMulIntegerStatus::invalid;
  int64_t reductionSize = type.getDimSize(1);
  return proveMatMulIntegerRanges(
      *elements, getMatMulIntegerInputRanges(input, reductionSize));
}

MatMulIntegerStatus getMatMulIntegerStatus(linalg::MatvecOp op) {
  auto inputs = op.getDpsInputs();
  auto inits = op.getDpsInits();
  auto weightType = inputs.size() == 2
                        ? dyn_cast<RankedTensorType>(inputs[0].getType())
                        : RankedTensorType{};
  if (!hasMatMulIntegerMarker(op) ||
      !isa<UnitAttr>(op->getAttr(CIMDialect::getMatMulIntegerAttrName())) ||
      inputs.size() != 2 || inits.size() != 1 || op->getNumResults() != 1 ||
      !weightType || weightType.getRank() != 2 ||
      weightType.getDimSize(0) <= 0 || weightType.getDimSize(1) <= 0)
    return MatMulIntegerStatus::invalid;
  int64_t outputSize = weightType.getDimSize(0);
  int64_t reductionSize = weightType.getDimSize(1);
  if (!hasType(inputs[0], {outputSize, reductionSize}, 8) ||
      !hasType(inputs[1], {reductionSize}, 8) ||
      !hasType(inits[0], {outputSize}, 32) ||
      !hasType(op->getResult(0), {outputSize}, 32) ||
      !isEvaluatedZeroSplat(inits[0]) ||
      !hasCanonicalInt8ContractionBody(op.getRegion(), 32) ||
      !hasCanonicalMatvecIndexingMaps(op))
    return MatMulIntegerStatus::invalid;
  return proveMatMulIntegerRanges(inputs[0], inputs[1]);
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
      !isa<UnitAttr>(op->getAttr(CIMDialect::getMatMulIntegerAttrName())) ||
      inputs.size() != 2 || inits.size() != 1 || op->getNumResults() != 1 ||
      !weightType || weightType.getRank() != 2 || !inputType ||
      inputType.getRank() != 2 || weightType.getDimSize(0) <= 0 ||
      weightType.getDimSize(1) <= 0 || inputType.getDimSize(1) <= 0)
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
      !hasCanonicalInt8ContractionBody(op.getRegion(), 32) ||
      !hasCanonicalMatmulIndexingMaps(op))
    return MatMulIntegerStatus::invalid;
  return proveMatMulIntegerRanges(inputs[0], inputs[1]);
}

bool isOffloadableRoot(Operation *op) {
  if (auto matvec = dyn_cast<linalg::MatvecOp>(op))
    return hasMatMulIntegerMarker(op)
               ? getMatMulIntegerStatus(matvec) == MatMulIntegerStatus::valid
               : isConvertible(matvec);
  if (auto matmul = dyn_cast<linalg::MatmulOp>(op))
    return hasMatMulIntegerMarker(op)
               ? getMatMulIntegerStatus(matmul) == MatMulIntegerStatus::valid
               : isConvertible(matmul);
  return false;
}

bool isPartitionedRoot(Operation *op) {
  auto segmentId =
      op->getAttrOfType<IntegerAttr>(CIMDialect::getSegmentIdAttrName());
  return segmentId && segmentId.getType().isSignlessInteger(64) &&
         segmentId.getInt() >= 0 && isOffloadableRoot(op);
}

LogicalResult rejectMatMulInteger(Operation *op, MatMulIntegerStatus status) {
  if (status == MatMulIntegerStatus::partialRangeOverflow)
    op->emitError(
        "ONNX MatMulInteger violates the signed i21 partial contract: "
        "partial is outside signed i21");
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
    if (!isPartitionedRoot(op) || !isConvertible(op))
      return rewriter.notifyMatchFailure(op, "not an exact CIM VMM candidate");

    auto inputs = adaptor.getInputs();
    int64_t segmentId =
        op->getAttrOfType<IntegerAttr>(CIMDialect::getSegmentIdAttrName())
            .getInt();
    auto weightType = cast<RankedTensorType>(inputs[0].getType());
    int64_t outputSize = weightType.getDimSize(0);
    int64_t reductionSize = weightType.getDimSize(1);
    Location location = op.getLoc();
    uint64_t tileCount =
        llvm::divideCeil(static_cast<uint64_t>(reductionSize), uint64_t{64});
    unsigned accumulationWidth = 21 + llvm::Log2_64_Ceil(tileCount);
    auto inputTileType =
        RankedTensorType::get({64}, weightType.getElementType());
    auto weightTileType =
        RankedTensorType::get({16, 64}, weightType.getElementType());
    auto resultType = cast<RankedTensorType>(op->getResult(0).getType());
    auto partialType = RankedTensorType::get({16}, resultType.getElementType());
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
        Value inputTile = inputs[1];
        if (reductionOffset != 0 || tileReductionSize != reductionSize)
          inputTile = tensor::ExtractSliceOp::create(
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
        Value weightTile = inputs[0];
        if (outputOffset != 0 || tileOutputSize != outputSize ||
            reductionOffset != 0 || tileReductionSize != reductionSize)
          weightTile = tensor::ExtractSliceOp::create(
              rewriter, location, weightSliceType, inputs[0], weightOffsets,
              weightSizes, weightStrides);
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
          weightTile = tensor::PadOp::create(rewriter, location, weightTileType,
                                             weightTile, low, high, zero,
                                             /*nofold=*/false);
        }

        Value partial = createIdentifiedVMM(
            rewriter, location, partialType, inputTile, weightTile, segmentId,
            getMatTile(op), outputOffset / kOutputTileSize,
            static_cast<int64_t>(reductionTile));
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
};

class FormMatmul final : public OpConversionPattern<linalg::MatmulOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::MatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isPartitionedRoot(op))
      return rewriter.notifyMatchFailure(op, "not a partitioned CIM root");
    bool integerProfile = hasMatMulIntegerMarker(op);
    if (integerProfile) {
      MatMulIntegerStatus status = getMatMulIntegerStatus(op);
      if (status != MatMulIntegerStatus::valid)
        return rejectMatMulInteger(op, status);
    } else if (!isConvertible(op)) {
      return rewriter.notifyMatchFailure(op,
                                         "not an exact CIM MatMul candidate");
    }

    Location location = op.getLoc();
    auto inputs = adaptor.getInputs();
    IntegerAttr segmentId =
        op->getAttrOfType<IntegerAttr>(CIMDialect::getSegmentIdAttrName());
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
      matvec->setAttr(CIMDialect::getSegmentIdAttrName(), segmentId);
      matvec->setAttr(CIMDialect::getMatTileAttrName(),
                      rewriter.getI64IntegerAttr(column));
      if (integerProfile) {
        matvec->setAttr(CIMDialect::getMatMulIntegerAttrName(),
                        rewriter.getUnitAttr());
      }
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
    if (!isPartitionedRoot(op) || !hasMatMulIntegerMarker(op))
      return rewriter.notifyMatchFailure(op,
                                         "not an ONNX MatMulInteger matvec");

    MatMulIntegerStatus status = getMatMulIntegerStatus(op);
    if (status != MatMulIntegerStatus::valid)
      return rejectMatMulInteger(op, status);

    Location location = op.getLoc();
    auto inputs = adaptor.getInputs();
    int64_t segmentId =
        op->getAttrOfType<IntegerAttr>(CIMDialect::getSegmentIdAttrName())
            .getInt();
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
            rewriter, location, partialType, inputTile, weightTile, segmentId,
            matTile, outputOffset / kOutputTileSize,
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

LogicalResult normalizeCIMConv(linalg::Conv2DNchwFchwOp op,
                               PatternRewriter &rewriter) {
  Location location = op.getLoc();
  auto reject = [&](StringRef message) {
    emitError(location) << "invalid marked ConvInteger: " << message;
    return failure();
  };
  if (!isa<UnitAttr>(op->getAttr(CIMDialect::getConvIntegerAttrName())))
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
  int64_t kernelHeight = weightType.getDimSize(2);
  int64_t kernelWidth = weightType.getDimSize(3);
  int64_t inputHeight = inputType.getDimSize(2);
  int64_t inputWidth = inputType.getDimSize(3);
  int64_t height = resultType.getDimSize(2);
  int64_t width = resultType.getDimSize(3);
  int64_t patchSize = 0;
  int64_t spatialSize = 0;
  if (channels <= 0 || filters <= 0 || kernelHeight <= 0 || kernelWidth <= 0 ||
      inputHeight <= 0 || inputWidth <= 0 || height <= 0 || width <= 0 ||
      llvm::MulOverflow(channels, kernelHeight, patchSize) ||
      llvm::MulOverflow(patchSize, kernelWidth, patchSize) ||
      llvm::MulOverflow(height, width, spatialSize) ||
      !hasType(inputs[0], {1, channels, inputHeight, inputWidth}, 8) ||
      !hasType(inputs[1], {filters, channels, kernelHeight, kernelWidth}, 8) ||
      !hasType(inits[0], {1, filters, height, width}, 32) ||
      !hasType(op->getResult(0), {1, filters, height, width}, 32) ||
      !isZeroSplat(inits[0]) || failed(evaluateDenseTensor(inputs[1])) ||
      !llvm::all_of(op.getStrides().getValues<int64_t>(),
                    [](int64_t value) { return value > 0; }) ||
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
      colType.getShape() != ArrayRef<int64_t>{1, patchSize, spatialSize} ||
      batchedResultType.getShape() !=
          ArrayRef<int64_t>{1, filters, spatialSize})
    return reject("upstream im2col contraction has unexpected batch shape");

  rewriter.setInsertionPoint(contraction);
  SmallVector<ReassociationIndices> collapseBatch{{0, 1}, {2}};
  auto col2DType =
      RankedTensorType::get({patchSize, spatialSize}, colType.getElementType());
  auto result2DType = RankedTensorType::get({filters, spatialSize},
                                            batchedResultType.getElementType());
  auto col2D = tensor::CollapseShapeOp::create(
      rewriter, location, col2DType, contractionInputs[1], collapseBatch);
  auto init2D = tensor::CollapseShapeOp::create(
      rewriter, location, result2DType, contractionInits[0], collapseBatch);
  auto matmul = linalg::MatmulOp::create(
      rewriter, location, TypeRange{result2DType},
      ValueRange{contractionInputs[0], col2D.getResult()},
      ValueRange{init2D.getResult()});
  matmul->setAttr(CIMDialect::getMatMulIntegerAttrName(),
                  rewriter.getUnitAttr());
  auto expanded =
      tensor::ExpandShapeOp::create(rewriter, location, batchedResultType,
                                    matmul.getResult(0), collapseBatch);
  rewriter.replaceOp(contraction, expanded.getResult());
  return success();
}

std::optional<std::pair<linalg::MatmulOp, bool>>
matchBiasMatmulProducer(Value output) {
  if (!output.hasOneUse())
    return std::nullopt;

  if (auto transpose = output.getDefiningOp<linalg::TransposeOp>()) {
    if (transpose.getPermutation() != ArrayRef<int64_t>{1, 0})
      return std::nullopt;
    auto matmul = transpose.getInput().getDefiningOp<linalg::MatmulOp>();
    auto sourceType =
        dyn_cast<RankedTensorType>(transpose.getInput().getType());
    auto resultType = dyn_cast<RankedTensorType>(output.getType());
    if (!matmul || !matmul.getResult(0).hasOneUse() || !sourceType ||
        !resultType || sourceType.getRank() != 2 || resultType.getRank() != 2 ||
        sourceType.getDimSize(0) != resultType.getDimSize(1) ||
        sourceType.getDimSize(1) != resultType.getDimSize(0))
      return std::nullopt;
    return std::pair<linalg::MatmulOp, bool>{matmul, false};
  }

  auto outer = output.getDefiningOp<tensor::ExpandShapeOp>();
  auto inner = outer ? outer.getSrc().getDefiningOp<tensor::ExpandShapeOp>()
                     : tensor::ExpandShapeOp{};
  auto matmul = inner ? inner.getSrc().getDefiningOp<linalg::MatmulOp>()
                      : linalg::MatmulOp{};
  if (!outer || !inner || !matmul || !inner.getResult().hasOneUse() ||
      !matmul.getResult(0).hasOneUse())
    return std::nullopt;

  auto matmulType = dyn_cast<RankedTensorType>(matmul.getResult(0).getType());
  auto innerType = dyn_cast<RankedTensorType>(inner.getResult().getType());
  auto outerType = dyn_cast<RankedTensorType>(outer.getResult().getType());
  int64_t spatialSize = 0;
  if (!matmulType || !innerType || !outerType || matmulType.getRank() != 2 ||
      innerType.getRank() != 3 || outerType.getRank() != 4 ||
      outerType.getDimSize(0) != 1 || outerType.getDimSize(2) <= 0 ||
      outerType.getDimSize(3) <= 0 ||
      llvm::MulOverflow(outerType.getDimSize(2), outerType.getDimSize(3),
                        spatialSize) ||
      innerType.getShape() != ArrayRef<int64_t>{1, matmulType.getDimSize(0),
                                                matmulType.getDimSize(1)} ||
      outerType.getDimSize(1) != matmulType.getDimSize(0) ||
      spatialSize != matmulType.getDimSize(1))
    return std::nullopt;
  return std::pair<linalg::MatmulOp, bool>{matmul, true};
}

class FoldCIMInt8BiasPattern final : public OpRewritePattern<linalg::AddOp> {
public:
  FoldCIMInt8BiasPattern(MLIRContext *context, bool allowExtraKTile)
      : OpRewritePattern(context), allowExtraKTile(allowExtraKTile) {}

  LogicalResult matchAndRewrite(linalg::AddOp add,
                                PatternRewriter &rewriter) const override {
    auto inputs = add.getDpsInputs();
    auto inits = add.getDpsInits();
    if (inputs.size() != 2 || inits.size() != 1 || add->getNumResults() != 1)
      return failure();

    auto broadcast = inputs[1].getDefiningOp<linalg::BroadcastOp>();
    auto empty = broadcast
                     ? broadcast.getInit().getDefiningOp<tensor::EmptyOp>()
                     : tensor::EmptyOp{};
    auto bias = broadcast
                    ? broadcast.getInput().getDefiningOp<arith::ConstantOp>()
                    : arith::ConstantOp{};
    auto biasElements = bias ? dyn_cast<DenseElementsAttr>(bias.getValue())
                             : DenseElementsAttr{};
    auto biasType =
        bias ? dyn_cast<RankedTensorType>(bias.getType()) : RankedTensorType{};
    if (!broadcast || !empty || !bias || !biasElements || !biasType ||
        biasType.getRank() != 1 || !biasType.getElementType().isInteger(32) ||
        broadcast.getResult().size() != 1 ||
        broadcast.getResult().front() != inputs[1] ||
        !broadcast.getResult().front().hasOneUse() ||
        !broadcast.getInput().hasOneUse() || inits[0] != empty.getResult())
      return failure();
    if (!llvm::all_of(empty.getResult().getUsers(), [&](Operation *user) {
          return user == add.getOperation() || user == broadcast.getOperation();
        }))
      return failure();

    auto producer = matchBiasMatmulProducer(inputs[0]);
    if (!producer)
      return failure();
    auto [matmul, convolution] = *producer;
    SmallVector<int64_t> expectedDimensions =
        convolution ? SmallVector<int64_t>{0, 2, 3} : SmallVector<int64_t>{0};
    if (broadcast.getDimensions() != ArrayRef<int64_t>(expectedDimensions) ||
        getMatMulIntegerStatus(matmul) != MatMulIntegerStatus::valid)
      return failure();

    auto matmulInputs = matmul.getDpsInputs();
    auto weightType = cast<RankedTensorType>(matmulInputs[0].getType());
    auto inputType = cast<RankedTensorType>(matmulInputs[1].getType());
    int64_t outputSize = weightType.getDimSize(0);
    int64_t reductionSize = weightType.getDimSize(1);
    int64_t columnCount = inputType.getDimSize(1);
    bool exactK = reductionSize % kReductionTileSize == 0;
    if (biasType.getDimSize(0) != outputSize ||
        biasElements.getNumElements() != outputSize ||
        (exactK && !allowExtraKTile))
      return failure();

    SmallVector<APInt> biasValues(biasElements.getValues<APInt>());
    if (!llvm::all_of(biasValues,
                      [](const APInt &value) { return value.isSignedIntN(8); }))
      return failure();

    FailureOr<DenseElementsAttr> weightElements =
        evaluateDenseTensor(matmulInputs[0]);
    if (failed(weightElements))
      return failure();
    SmallVector<APInt> weightValues((*weightElements).getValues<APInt>());
    int64_t biasLane = exactK ? reductionSize - 1 : reductionSize;
    SmallVector<APInt> newWeightValues;
    newWeightValues.reserve(outputSize * (reductionSize + 1));
    for (int64_t output = 0; output < outputSize; ++output) {
      for (int64_t reduction = 0; reduction < reductionSize; ++reduction) {
        if (reduction == biasLane)
          newWeightValues.push_back(biasValues[output].sextOrTrunc(8));
        newWeightValues.push_back(
            weightValues[output * reductionSize + reduction]);
      }
      if (!exactK)
        newWeightValues.push_back(biasValues[output].sextOrTrunc(8));
    }

    auto newWeightType = RankedTensorType::get({outputSize, reductionSize + 1},
                                               weightType.getElementType());
    auto newWeightElements =
        DenseElementsAttr::get(newWeightType, newWeightValues);
    SmallVector<IntegerRange> inputRanges(reductionSize + 1, {-128, 127});
    inputRanges[biasLane] = {1, 1};
    if (proveMatMulIntegerRanges(newWeightElements, inputRanges) !=
        MatMulIntegerStatus::valid)
      return failure();

    Location location = matmul.getLoc();
    rewriter.setInsertionPoint(matmul);
    Value newWeight = arith::ConstantOp::create(
        rewriter, location, newWeightType, newWeightElements);
    auto oneType =
        RankedTensorType::get({1, columnCount}, inputType.getElementType());
    Value one = arith::ConstantOp::create(
        rewriter, location, oneType,
        DenseElementsAttr::get(
            oneType, rewriter.getIntegerAttr(inputType.getElementType(), 1)));

    SmallVector<Value> inputParts;
    if (exactK) {
      auto prefixType = RankedTensorType::get({reductionSize - 1, columnCount},
                                              inputType.getElementType());
      auto lastType =
          RankedTensorType::get({1, columnCount}, inputType.getElementType());
      SmallVector<OpFoldResult> prefixOffsets{rewriter.getIndexAttr(0),
                                              rewriter.getIndexAttr(0)};
      SmallVector<OpFoldResult> prefixSizes{
          rewriter.getIndexAttr(reductionSize - 1),
          rewriter.getIndexAttr(columnCount)};
      SmallVector<OpFoldResult> lastOffsets{
          rewriter.getIndexAttr(reductionSize - 1), rewriter.getIndexAttr(0)};
      SmallVector<OpFoldResult> lastSizes{rewriter.getIndexAttr(1),
                                          rewriter.getIndexAttr(columnCount)};
      SmallVector<OpFoldResult> strides{rewriter.getIndexAttr(1),
                                        rewriter.getIndexAttr(1)};
      Value prefix = tensor::ExtractSliceOp::create(
          rewriter, location, prefixType, matmulInputs[1], prefixOffsets,
          prefixSizes, strides);
      Value last = tensor::ExtractSliceOp::create(rewriter, location, lastType,
                                                  matmulInputs[1], lastOffsets,
                                                  lastSizes, strides);
      inputParts = {prefix, one, last};
    } else {
      inputParts = {matmulInputs[1], one};
    }
    Value newInput =
        tensor::ConcatOp::create(rewriter, location, 0, inputParts);
    auto newMatmul = linalg::MatmulOp::create(
        rewriter, location, TypeRange{matmul.getResult(0).getType()},
        ValueRange{newWeight, newInput}, matmul.getDpsInits());
    newMatmul->setAttr(CIMDialect::getMatMulIntegerAttrName(),
                       rewriter.getUnitAttr());

    rewriter.replaceOp(matmul, newMatmul.getResults());
    rewriter.replaceOp(add, inputs[0]);
    rewriter.eraseOp(broadcast);
    rewriter.eraseOp(bias);
    rewriter.eraseOp(empty);
    return success();
  }

private:
  bool allowExtraKTile;
};

class NormalizeCIMConv final
    : public impl::NormalizeCIMConvBase<NormalizeCIMConv> {
public:
  using Base::Base;

  void runOnOperation() override {
    SmallVector<linalg::Conv2DNchwFchwOp> candidates;
    getOperation()->walk([&](linalg::Conv2DNchwFchwOp op) {
      if (op->hasAttr(CIMDialect::getConvIntegerAttrName()))
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

class FoldCIMInt8Bias final
    : public impl::FoldCIMInt8BiasBase<FoldCIMInt8Bias> {
public:
  using Base::Base;

  void runOnOperation() override {
    if (!enable)
      return;
    RewritePatternSet patterns(&getContext());
    patterns.add<FoldCIMInt8BiasPattern>(&getContext(), allowExtraKTile);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

class VerifyCIMExecutionPlan final
    : public impl::VerifyCIMExecutionPlanBase<VerifyCIMExecutionPlan> {
public:
  using Base::Base;

  void runOnOperation() override {
    if (failed(verifyCIMExecutionPlan(getOperation())))
      signalPassFailure();
  }
};

class PartitionCIMProgramPass final
    : public impl::PartitionCIMProgramBase<PartitionCIMProgramPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    Builder builder(&getContext());
    for (func::FuncOp function : getOperation().getOps<func::FuncOp>()) {
      int64_t segmentId = 0;
      function.walk([&](Operation *op) {
        if (!isOffloadableRoot(op))
          return;
        op->setAttr(CIMDialect::getSegmentIdAttrName(),
                    builder.getI64IntegerAttr(segmentId++));
      });
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
      return !op->hasAttr(CIMDialect::getSegmentIdAttrName());
    });
    target.addDynamicallyLegalOp<linalg::MatmulOp>([](linalg::MatmulOp op) {
      return !op->hasAttr(CIMDialect::getSegmentIdAttrName());
    });
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    RewritePatternSet patterns(&getContext());
    patterns.add<FormMatvec, FormMatmul, FormMatMulIntegerMatvec>(
        &getContext());
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
    SmallVector<VMMOp> allVMMs;
    function.walk([&](VMMOp op) { allVMMs.push_back(op); });
    if (allVMMs.empty())
      return;

    if (!function.getBody().hasOneBlock()) {
      function.emitError(
          "materialize-cim-schedule requires one straight-line block");
      return signalPassFailure();
    }
    Block *body = &function.getBody().front();
    if (llvm::any_of(allVMMs,
                     [body](VMMOp op) { return op->getBlock() != body; })) {
      function.emitError(
          "materialize-cim-schedule cannot represent nested control flow");
      return signalPassFailure();
    }

    SmallVector<SmallVector<VMMOp>> segments;
    for (VMMOp vmm : allVMMs) {
      auto segmentId =
          readNonNegativeI64(vmm, CIMDialect::getSegmentIdAttrName());
      if (!segmentId)
        return signalPassFailure();
      if (segments.empty() ||
          *segmentId != static_cast<int64_t>(segments.size() - 1)) {
        if (*segmentId != static_cast<int64_t>(segments.size())) {
          vmm.emitOpError("materialize-cim-schedule requires dense, "
                          "contiguous cim.segment_id ordering");
          return signalPassFailure();
        }
        segments.emplace_back();
      }
      segments.back().push_back(vmm);
    }

    for (SmallVector<VMMOp> &vmms : segments) {
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

      int64_t mTiles = 0;
      int64_t nTiles = 0;
      int64_t kTiles = 0;
      if (llvm::AddOverflow(maxM, int64_t{1}, mTiles) ||
          llvm::AddOverflow(maxN, int64_t{1}, nTiles) ||
          llvm::AddOverflow(maxK, int64_t{1}, kTiles)) {
        function.emitError(
            "materialize-cim-schedule tile extent overflows i64");
        return signalPassFailure();
      }
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
          vmms[index].emitOpError(
              "materialize-cim-schedule requires contiguous "
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
        // TODO(CTQ-031): Keep software-only groups two-wide until cross-core
        // waves are known.
        int64_t groupId = workId / 2;
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
  }
};

const llvm::StringRef kExecutionPlanIdentityAttrs[] = {
    CIMDialect::getSegmentIdAttrName(),
    "m_tile",
    "n_tile",
    "k_tile",
    "work_id",
    "group_id",
    "core_slot",
    "macro_slot",
    "cim.mapping"};
constexpr int64_t kOutputCacheRows = 8;

void copyExecutionPlanIdentity(Operation *from, Operation *to) {
  for (StringRef name : kExecutionPlanIdentityAttrs)
    to->setAttr(name, from->getAttr(name));
}

bool isExecutionPlanOp(Operation *op) {
  return isa<ConfigureInputOp, ConfigureWeightOp, DispatchOp, OnceOp,
             ReadbackOp, GroupBarrierOp>(op);
}

class MaterializeCIMExecutionPlan final
    : public impl::MaterializeCIMExecutionPlanBase<
          MaterializeCIMExecutionPlan> {
public:
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    struct WorkPlan {
      VMMOp vmm;
      DenseIntElementsAttr weight;
      int64_t segmentId;
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
      bool hasExecutionPlan = false;
      function.walk(
          [&](Operation *op) { hasExecutionPlan |= isExecutionPlanOp(op); });
      if (vmms.empty()) {
        if (hasExecutionPlan && failed(verifyCIMExecutionPlan(function)))
          return signalPassFailure();
        continue;
      }
      if (hasExecutionPlan) {
        function.emitError("materialize-cim-execution-plan rejects mixed VMM "
                           "and execution-plan ops");
        return signalPassFailure();
      }
      if (!function.getBody().hasOneBlock()) {
        function.emitError(
            "materialize-cim-execution-plan requires one straight-line block");
        return signalPassFailure();
      }
      if (function->hasAttr("cim.execution_plan_schema_version")) {
        function.emitError(
            "materialize-cim-execution-plan rejects a stale execution-plan "
            "schema attribute");
        return signalPassFailure();
      }

      FunctionPlan plan{function, {}};
      plan.works.reserve(vmms.size());
      int64_t currentSegment = -1;
      int64_t expectedWork = 0;
      for (VMMOp vmm : vmms) {
        if (countPresent(vmm, kExecutionPlanIdentityAttrs) !=
            std::size(kExecutionPlanIdentityAttrs)) {
          vmm.emitOpError(
              "materialize-cim-execution-plan requires complete mapped "
              "work identity");
          return signalPassFailure();
        }
        auto segmentId =
            readNonNegativeI64(vmm, CIMDialect::getSegmentIdAttrName());
        auto workId = readNonNegativeI64(vmm, "work_id");
        auto groupId = readNonNegativeI64(vmm, "group_id");
        auto coreSlot = readNonNegativeI64(vmm, "core_slot");
        auto macroSlot = readNonNegativeI64(vmm, "macro_slot");
        if (!segmentId || !workId || !groupId || !coreSlot || !macroSlot)
          return signalPassFailure();
        if (*segmentId != currentSegment) {
          if (*segmentId != currentSegment + 1) {
            vmm.emitOpError("materialize-cim-execution-plan requires dense, "
                            "contiguous cim.segment_id ordering");
            return signalPassFailure();
          }
          currentSegment = *segmentId;
          expectedWork = 0;
        }
        if (*workId != expectedWork || *groupId != *workId / 2 ||
            *macroSlot > 1) {
          vmm.emitOpError(
              "materialize-cim-execution-plan rejects invalid work placement");
          return signalPassFailure();
        }
        ++expectedWork;
        FailureOr<DenseElementsAttr> weight =
            evaluateDenseTensor(vmm.getWeight());
        if (failed(weight) ||
            (*weight).getType() != vmm.getWeight().getType()) {
          vmm.emitOpError(
              "materialize-cim-execution-plan requires a compile-time "
              "constant weight tile");
          return signalPassFailure();
        }
        auto intWeight = dyn_cast<DenseIntElementsAttr>(*weight);
        if (!intWeight) {
          vmm.emitOpError("materialize-cim-execution-plan requires INT8 weight "
                          "elements");
          return signalPassFailure();
        }
        plan.works.push_back({vmm, intWeight, *segmentId, *workId, *groupId,
                              *coreSlot, *macroSlot});
      }

      for (size_t index = 0; index < plan.works.size();) {
        WorkPlan &first = plan.works[index];
        if (index + 1 == plan.works.size() ||
            plan.works[index + 1].segmentId != first.segmentId) {
          ++index;
          continue;
        }
        WorkPlan &second = plan.works[index + 1];
        if (first.groupId != second.groupId ||
            first.coreSlot != second.coreSlot || first.macroSlot != 0 ||
            second.macroSlot != 1 ||
            first.vmm->getAttr("cim.mapping") !=
                second.vmm->getAttr("cim.mapping")) {
          second.vmm.emitOpError(
              "materialize-cim-execution-plan requires a same-route Macro 0/1 "
              "pair per two-work group");
          return signalPassFailure();
        }
        index += 2;
      }

      for (size_t begin = 0; begin < plan.works.size();) {
        size_t end = begin + 1;
        while (end < plan.works.size() &&
               plan.works[end].segmentId == plan.works[begin].segmentId)
          ++end;

        llvm::DenseSet<Operation *> segmentWork;
        for (size_t index = begin; index < end; ++index)
          segmentWork.insert(plan.works[index].vmm);

        llvm::SetVector<Operation *> preparation;
        BackwardSliceOptions sliceOptions;
        sliceOptions.inclusive = true;
        sliceOptions.omitUsesFromAbove = false;
        for (size_t index = begin; index < end; ++index) {
          Operation *definition =
              plan.works[index].vmm.getInput().getDefiningOp();
          if (!definition)
            continue;
          preparation.insert(definition);
          llvm::SetVector<Operation *> slice;
          if (failed(getBackwardSlice(definition, &slice, sliceOptions))) {
            definition->emitError("materialize-cim-execution-plan cannot "
                                  "analyze segment input preparation");
            return signalPassFailure();
          }
          preparation.insert(slice.begin(), slice.end());
        }
        // Configure inputs must be materialized before this segment starts;
        // hoist their pure preparation when the source IR placed it later.
        if (llvm::any_of(preparation, [&](Operation *op) {
              return segmentWork.contains(op);
            })) {
          plan.works[begin].vmm.emitOpError(
              "materialize-cim-execution-plan rejects an input that depends "
              "on work in the same segment");
          return signalPassFailure();
        }
        Operation *segmentStart = plan.works[begin].vmm;
        SmallVector<Operation *> toMove;
        for (Operation *op : preparation) {
          if (op->getBlock() != segmentStart->getBlock() ||
              !segmentStart->isBeforeInBlock(op))
            continue;
          if (!isMemoryEffectFree(op)) {
            op->emitError("materialize-cim-execution-plan cannot move "
                          "side-effecting segment input preparation");
            return signalPassFailure();
          }
          toMove.push_back(op);
        }
        llvm::sort(toMove, [](Operation *lhs, Operation *rhs) {
          return lhs->isBeforeInBlock(rhs);
        });
        for (Operation *op : toMove)
          op->moveBefore(segmentStart);
        begin = end;
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
            (Twine("__cim_weight_") + plan.function.getSymName() + "_s" +
             Twine(work.segmentId) + "_w" + Twine(work.workId))
                .str();
        StaticWeightOp::create(moduleBuilder, work.vmm.getLoc(), name,
                               work.weight);
        resources.push_back(FlatSymbolRefAttr::get(module.getContext(), name));
      }

      plan.function->setAttr("cim.execution_plan_schema_version",
                             moduleBuilder.getI64IntegerAttr(1));
      for (size_t index = 0; index < plan.works.size();) {
        size_t count =
            index + 1 < plan.works.size() && plan.works[index + 1].segmentId ==
                                                 plan.works[index].segmentId
                ? 2
                : 1;
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
          copyExecutionPlanIdentity(work.vmm, input);
          auto weight = ConfigureWeightOp::create(builder, work.vmm.getLoc(),
                                                  resources[index + offset]);
          copyExecutionPlanIdentity(work.vmm, weight);
        }
        for (size_t offset = 0; offset < count; ++offset) {
          WorkPlan &work = plan.works[index + offset];
          auto dispatch = DispatchOp::create(builder, work.vmm.getLoc());
          copyExecutionPlanIdentity(work.vmm, dispatch);
        }
        WorkPlan &first = plan.works[index];
        auto once = OnceOp::create(builder, first.vmm.getLoc());
        once->setAttr(CIMDialect::getSegmentIdAttrName(),
                      first.vmm->getAttr(CIMDialect::getSegmentIdAttrName()));
        once->setAttr("group_id", first.vmm->getAttr("group_id"));
        once->setAttr("core_slot", first.vmm->getAttr("core_slot"));
        once->setAttr("cim.mapping", first.vmm->getAttr("cim.mapping"));
        for (size_t offset = 0; offset < count; ++offset) {
          WorkPlan &work = plan.works[index + offset];
          auto readback = ReadbackOp::create(builder, work.vmm.getLoc(),
                                             work.vmm.getResult().getType());
          copyExecutionPlanIdentity(work.vmm, readback);
          auto nTile = readNonNegativeI64(work.vmm, "n_tile");
          if (!nTile) {
            work.vmm.emitOpError(
                "materialize-cim-execution-plan requires n_tile for the "
                "INT8 output Cache");
            return signalPassFailure();
          }
          auto mapping = work.vmm->getAttrOfType<DictionaryAttr>("cim.mapping");
          auto route = mapping ? mapping.getAs<DenseI64ArrayAttr>("route")
                               : DenseI64ArrayAttr();
          if (!route || route.size() != 6) {
            work.vmm.emitOpError(
                "materialize-cim-execution-plan requires a six-element "
                "mapped route for readback");
            return signalPassFailure();
          }
          // Groups are read back before the next group starts, so the eight
          // output-cache rows can be reused by successive N tiles.
          readback->setAttr(
              "output_cache_address",
              builder.getI64IntegerAttr(*nTile % kOutputCacheRows));
          readback->setAttr("test_core_xy",
                            builder.getI64IntegerAttr(-route[0]));
          readback->setAttr("test_core_x",
                            builder.getI64IntegerAttr(-route[1]));
          readback->setAttr("test_core_y",
                            builder.getI64IntegerAttr(-route[2]));
          readbacks.push_back(readback);
        }
        auto barrier = GroupBarrierOp::create(builder, first.vmm.getLoc());
        barrier->setAttr(
            CIMDialect::getSegmentIdAttrName(),
            first.vmm->getAttr(CIMDialect::getSegmentIdAttrName()));
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
        index += count;
      }
      if (failed(verifyCIMExecutionPlan(plan.function)))
        return signalPassFailure();
    }
  }
};
} // namespace
} // namespace mlir::cim
