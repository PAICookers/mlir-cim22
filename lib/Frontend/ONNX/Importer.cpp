//===- Importer.cpp - Quantized ONNX importer -------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIM22/Frontend/ONNX/Importer.h"

#include "onnx/onnx_pb.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

#include <fstream>
#include <limits>

namespace mlir::cim {
namespace {
constexpr StringLiteral kMatMulIntegerMarker = "cim.onnx.matmul_integer";
constexpr StringLiteral kConvIntegerMarker = "cim.onnx.conv_integer";

struct MatMulIntegerModel {
  int64_t batch;
  int64_t reduction;
  int64_t output;
  SmallVector<APInt> weightValues;
};

struct ConvIntegerModel {
  int64_t channels;
  int64_t height;
  int64_t width;
  int64_t filters;
  SmallVector<APInt> weightValues;
};

template <typename T>
FailureOr<T> reject(MLIRContext &context, StringRef reason) {
  emitError(UnknownLoc::get(&context)) << "quantized ONNX importer: " << reason;
  return failure();
}

LogicalResult reject(MLIRContext &context, StringRef reason) {
  emitError(UnknownLoc::get(&context)) << "quantized ONNX importer: " << reason;
  return failure();
}

FailureOr<SmallVector<int64_t>>
getStaticTensorShape(MLIRContext &context, const onnx::ValueInfoProto &value,
                     int elementType, StringRef role) {
  if (!value.has_type() || !value.type().has_tensor_type())
    return reject<SmallVector<int64_t>>(context,
                                        (role + " must be a tensor").str());

  const auto &tensor = value.type().tensor_type();
  if (tensor.elem_type() != elementType || !tensor.has_shape())
    return reject<SmallVector<int64_t>>(
        context, (role + " has an unsupported element type or shape").str());

  SmallVector<int64_t> shape;
  shape.reserve(tensor.shape().dim_size());
  for (const auto &dimension : tensor.shape().dim()) {
    if (!dimension.has_dim_value() || dimension.dim_value() <= 0)
      return reject<SmallVector<int64_t>>(
          context, (role + " must have positive static dimensions").str());
    shape.push_back(dimension.dim_value());
  }
  return shape;
}

FailureOr<SmallVector<APInt>>
extractInt8Initializer(MLIRContext &context, const onnx::TensorProto &tensor,
                       int64_t elementCount, StringRef role = "weight") {
  if (tensor.data_type() != onnx::TensorProto::INT8 ||
      tensor.data_location() != onnx::TensorProto::DEFAULT ||
      tensor.external_data_size() != 0)
    return reject<SmallVector<APInt>>(
        context, (role + " must be an embedded INT8 initializer").str());

  SmallVector<APInt> values;
  values.reserve(elementCount);
  if (!tensor.raw_data().empty()) {
    if (tensor.raw_data().size() != static_cast<size_t>(elementCount))
      return reject<SmallVector<APInt>>(
          context, (role + " raw_data size does not match its shape").str());
    for (unsigned char byte : tensor.raw_data())
      values.emplace_back(8, byte);
    return values;
  }

  if (tensor.int32_data_size() != elementCount)
    return reject<SmallVector<APInt>>(
        context,
        (role + " must use raw_data or one int32_data value per element")
            .str());
  for (int32_t value : tensor.int32_data()) {
    if (value < -128 || value > 127)
      return reject<SmallVector<APInt>>(
          context, (role + " INT8 int32_data value is out of range").str());
    values.emplace_back(8, static_cast<uint64_t>(value), true);
  }
  return values;
}

bool hasIntValues(const onnx::AttributeProto &attribute,
                  ArrayRef<int64_t> expected) {
  return attribute.type() == onnx::AttributeProto::INTS &&
         llvm::equal(attribute.ints(), expected);
}

LogicalResult validateConvAttributes(MLIRContext &context,
                                     const onnx::NodeProto &node) {
  bool hasPads = false;
  for (const onnx::AttributeProto &attribute : node.attribute()) {
    StringRef name = attribute.name();
    if (name == "pads") {
      hasPads = hasIntValues(attribute, {1, 1, 1, 1});
      if (!hasPads)
        return reject(context, "ConvInteger requires pads=[1,1,1,1]");
      continue;
    }
    if (name == "strides" && hasIntValues(attribute, {1, 1}))
      continue;
    if (name == "dilations" && hasIntValues(attribute, {1, 1}))
      continue;
    if (name == "kernel_shape" && hasIntValues(attribute, {3, 3}))
      continue;
    if (name == "group" && attribute.type() == onnx::AttributeProto::INT &&
        attribute.i() == 1)
      continue;
    return reject(context, "ConvInteger has an unsupported attribute");
  }
  return hasPads ? success()
                 : reject(context, "ConvInteger requires explicit pad1");
}

LogicalResult validateZeroPoint(MLIRContext &context,
                                const onnx::TensorProto &tensor,
                                int64_t filters, bool perChannel,
                                StringRef role) {
  int64_t elementCount = 0;
  if (tensor.dims_size() == 0) {
    elementCount = 1;
  } else if (perChannel && tensor.dims_size() == 1 &&
             tensor.dims(0) == filters) {
    elementCount = filters;
  } else {
    return reject(context,
                  (role + " must be scalar or per-output-channel").str());
  }

  FailureOr<SmallVector<APInt>> values =
      extractInt8Initializer(context, tensor, elementCount, role);
  if (failed(values))
    return failure();
  if (!llvm::all_of(*values, [](const APInt &value) { return value.isZero(); }))
    return reject(context, (role + " must be zero").str());
  return success();
}

FailureOr<MatMulIntegerModel>
validateMatMulIntegerModel(MLIRContext &context,
                           const onnx::ModelProto &model) {
  unsigned defaultOpsetCount = 0;
  bool hasSupportedOpset = false;
  for (const onnx::OperatorSetIdProto &opset : model.opset_import()) {
    if (!opset.domain().empty())
      continue;
    ++defaultOpsetCount;
    hasSupportedOpset = opset.version() == 10;
  }
  if (defaultOpsetCount != 1 || !hasSupportedOpset)
    return reject<MatMulIntegerModel>(
        context, "requires the default ai.onnx opset at version 10");
  if (model.functions_size() != 0)
    return reject<MatMulIntegerModel>(context,
                                      "does not support model-local functions");

  const onnx::GraphProto &graph = model.graph();
  if (graph.node_size() != 1 || graph.output_size() != 1)
    return reject<MatMulIntegerModel>(
        context, "currently supports one MatMulInteger node and one output");

  llvm::DenseMap<StringRef, const onnx::TensorProto *> initializers;
  for (const onnx::TensorProto &initializer : graph.initializer()) {
    if (initializer.name().empty() ||
        !initializers.try_emplace(initializer.name(), &initializer).second)
      return reject<MatMulIntegerModel>(
          context, "initializer names must be non-empty and unique");
  }

  SmallVector<const onnx::ValueInfoProto *> runtimeInputs;
  llvm::DenseMap<StringRef, const onnx::ValueInfoProto *> graphInputs;
  for (const onnx::ValueInfoProto &input : graph.input()) {
    if (input.name().empty() ||
        !graphInputs.try_emplace(input.name(), &input).second)
      return reject<MatMulIntegerModel>(
          context, "graph input names must be non-empty and unique");
    if (!initializers.contains(input.name()))
      runtimeInputs.push_back(&input);
  }
  if (runtimeInputs.size() != 1)
    return reject<MatMulIntegerModel>(
        context, "currently supports one runtime activation input");

  const onnx::NodeProto &node = graph.node(0);
  if (node.domain() != "" || node.op_type() != "MatMulInteger" ||
      node.attribute_size() != 0 || node.input_size() != 2 ||
      node.output_size() != 1)
    return reject<MatMulIntegerModel>(
        context, "supports two-input MatMulInteger with omitted zero-points");
  if (node.input(0) != runtimeInputs.front()->name())
    return reject<MatMulIntegerModel>(
        context, "MatMulInteger A must be the runtime graph input");
  auto weightIt = initializers.find(node.input(1));
  if (weightIt == initializers.end())
    return reject<MatMulIntegerModel>(
        context, "MatMulInteger B must be an embedded initializer");
  if (node.output(0) != graph.output(0).name())
    return reject<MatMulIntegerModel>(
        context, "MatMulInteger output must be the graph output");

  FailureOr<SmallVector<int64_t>> inputShape = getStaticTensorShape(
      context, *runtimeInputs.front(), onnx::TensorProto::INT8, "activation");
  FailureOr<SmallVector<int64_t>> outputShape = getStaticTensorShape(
      context, graph.output(0), onnx::TensorProto::INT32, "result");
  if (failed(inputShape) || failed(outputShape))
    return failure();
  if (inputShape->size() != 2 || outputShape->size() != 2)
    return reject<MatMulIntegerModel>(
        context, "requires rank-2 activation and result tensors");

  const onnx::TensorProto &weight = *weightIt->second;
  if (weight.dims_size() != 2 || weight.dims(0) <= 0 || weight.dims(1) <= 0)
    return reject<MatMulIntegerModel>(
        context, "weight must have two positive static dimensions");
  int64_t batch = (*inputShape)[0];
  int64_t reduction = (*inputShape)[1];
  int64_t output = weight.dims(1);
  if (weight.dims(0) != reduction || (*outputShape)[0] != batch ||
      (*outputShape)[1] != output)
    return reject<MatMulIntegerModel>(
        context, "MatMulInteger shapes must satisfy [M,K] * [K,N] -> [M,N]");
  if (reduction > std::numeric_limits<int64_t>::max() / output)
    return reject<MatMulIntegerModel>(context,
                                      "weight element count is too large");

  FailureOr<SmallVector<APInt>> weightValues =
      extractInt8Initializer(context, weight, reduction * output);
  if (failed(weightValues))
    return failure();
  return MatMulIntegerModel{batch, reduction, output, std::move(*weightValues)};
}

FailureOr<ConvIntegerModel>
validateConvIntegerModel(MLIRContext &context, const onnx::ModelProto &model) {
  unsigned defaultOpsetCount = 0;
  bool hasSupportedOpset = false;
  for (const onnx::OperatorSetIdProto &opset : model.opset_import()) {
    if (!opset.domain().empty())
      continue;
    ++defaultOpsetCount;
    hasSupportedOpset = opset.version() == 10;
  }
  if (defaultOpsetCount != 1 || !hasSupportedOpset)
    return reject<ConvIntegerModel>(
        context, "requires the default ai.onnx opset at version 10");
  if (model.functions_size() != 0)
    return reject<ConvIntegerModel>(context,
                                    "does not support model-local functions");

  const onnx::GraphProto &graph = model.graph();
  if (graph.node_size() != 1 || graph.output_size() != 1)
    return reject<ConvIntegerModel>(
        context, "currently supports one ConvInteger node and one output");

  llvm::DenseMap<StringRef, const onnx::TensorProto *> initializers;
  for (const onnx::TensorProto &initializer : graph.initializer()) {
    if (initializer.name().empty() ||
        !initializers.try_emplace(initializer.name(), &initializer).second)
      return reject<ConvIntegerModel>(
          context, "initializer names must be non-empty and unique");
  }

  SmallVector<const onnx::ValueInfoProto *> runtimeInputs;
  llvm::DenseMap<StringRef, const onnx::ValueInfoProto *> graphInputs;
  for (const onnx::ValueInfoProto &input : graph.input()) {
    if (input.name().empty() ||
        !graphInputs.try_emplace(input.name(), &input).second)
      return reject<ConvIntegerModel>(
          context, "graph input names must be non-empty and unique");
    if (!initializers.contains(input.name()))
      runtimeInputs.push_back(&input);
  }
  if (runtimeInputs.size() != 1)
    return reject<ConvIntegerModel>(
        context, "currently supports one runtime activation input");

  const onnx::NodeProto &node = graph.node(0);
  if (!node.domain().empty() || node.op_type() != "ConvInteger" ||
      node.input_size() < 2 || node.input_size() > 4 || node.output_size() != 1)
    return reject<ConvIntegerModel>(
        context, "supports only the frozen ConvInteger profile");
  if (failed(validateConvAttributes(context, node)))
    return failure();
  if (node.input(0) != runtimeInputs.front()->name())
    return reject<ConvIntegerModel>(
        context, "ConvInteger input must be the runtime graph input");
  auto weightIt = initializers.find(node.input(1));
  if (weightIt == initializers.end())
    return reject<ConvIntegerModel>(
        context, "ConvInteger weight must be an embedded initializer");
  if (node.output(0) != graph.output(0).name())
    return reject<ConvIntegerModel>(
        context, "ConvInteger output must be the graph output");

  FailureOr<SmallVector<int64_t>> inputShape = getStaticTensorShape(
      context, *runtimeInputs.front(), onnx::TensorProto::INT8, "activation");
  FailureOr<SmallVector<int64_t>> outputShape = getStaticTensorShape(
      context, graph.output(0), onnx::TensorProto::INT32, "result");
  if (failed(inputShape) || failed(outputShape))
    return failure();
  if (inputShape->size() != 4 || outputShape->size() != 4 ||
      (*inputShape)[0] != 1 || (*outputShape)[0] != 1)
    return reject<ConvIntegerModel>(
        context, "ConvInteger requires static batch-one NCHW tensors");

  const onnx::TensorProto &weight = *weightIt->second;
  if (weight.dims_size() != 4 || weight.dims(0) <= 0 || weight.dims(1) <= 0 ||
      weight.dims(2) != 3 || weight.dims(3) != 3)
    return reject<ConvIntegerModel>(
        context, "ConvInteger weight must have shape [F,C,3,3]");
  int64_t channels = (*inputShape)[1];
  int64_t height = (*inputShape)[2];
  int64_t width = (*inputShape)[3];
  int64_t filters = weight.dims(0);
  if (weight.dims(1) != channels || (*outputShape)[1] != filters ||
      (*outputShape)[2] != height || (*outputShape)[3] != width)
    return reject<ConvIntegerModel>(
        context, "ConvInteger shapes do not match 3x3 stride1 pad1");
  if (height > std::numeric_limits<int64_t>::max() - 2 ||
      width > std::numeric_limits<int64_t>::max() - 2 ||
      channels > std::numeric_limits<int64_t>::max() / 9 ||
      filters > std::numeric_limits<int64_t>::max() / (channels * 9))
    return reject<ConvIntegerModel>(context, "ConvInteger shape is too large");

  if (node.input_size() >= 3 && !node.input(2).empty()) {
    auto zeroPoint = initializers.find(node.input(2));
    if (zeroPoint == initializers.end())
      return reject<ConvIntegerModel>(
          context, "activation zero-point must be an embedded initializer");
    if (failed(validateZeroPoint(context, *zeroPoint->second, filters,
                                 /*perChannel=*/false,
                                 "activation zero-point")))
      return failure();
  }
  if (node.input_size() == 4 && !node.input(3).empty()) {
    auto zeroPoint = initializers.find(node.input(3));
    if (zeroPoint == initializers.end())
      return reject<ConvIntegerModel>(
          context, "weight zero-point must be an embedded initializer");
    if (failed(validateZeroPoint(context, *zeroPoint->second, filters,
                                 /*perChannel=*/true, "weight zero-point")))
      return failure();
  }

  FailureOr<SmallVector<APInt>> weightValues =
      extractInt8Initializer(context, weight, filters * channels * 9, "weight");
  if (failed(weightValues))
    return failure();
  return ConvIntegerModel{channels, height, width, filters,
                          std::move(*weightValues)};
}

DenseElementsAttr transposeWeightConstant(Builder &builder,
                                          const MatMulIntegerModel &model) {
  auto elementType = builder.getIntegerType(8);
  auto type =
      RankedTensorType::get({model.output, model.reduction}, elementType);
  SmallVector<APInt> transposed;
  transposed.reserve(model.weightValues.size());
  for (int64_t output = 0; output < model.output; ++output)
    for (int64_t reduction = 0; reduction < model.reduction; ++reduction)
      transposed.push_back(
          model.weightValues[reduction * model.output + output]);
  return DenseElementsAttr::get(type, transposed);
}
} // namespace

FailureOr<OwningOpRef<ModuleOp>> importQuantizedONNX(MLIRContext &context,
                                                     StringRef filename) {
  std::ifstream input(filename.str(), std::ios::binary);
  if (!input)
    return reject<OwningOpRef<ModuleOp>>(context, "cannot open ONNX model");

  onnx::ModelProto model;
  if (!model.ParseFromIstream(&input))
    return reject<OwningOpRef<ModuleOp>>(context,
                                         "cannot parse binary ONNX ModelProto");
  return importQuantizedONNX(context, model);
}

static FailureOr<OwningOpRef<ModuleOp>>
importMatMulInteger(MLIRContext &context, const onnx::ModelProto &model) {
  FailureOr<MatMulIntegerModel> imported =
      validateMatMulIntegerModel(context, model);
  if (failed(imported))
    return failure();

  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<func::FuncDialect>();
  context.getOrLoadDialect<linalg::LinalgDialect>();
  context.getOrLoadDialect<tensor::TensorDialect>();

  Location location = UnknownLoc::get(&context);
  auto i8 = IntegerType::get(&context, 8);
  auto i32 = IntegerType::get(&context, 32);
  auto onnxInputType =
      RankedTensorType::get({imported->batch, imported->reduction}, i8);
  auto weightType =
      RankedTensorType::get({imported->output, imported->reduction}, i8);
  auto normalizedResultType =
      RankedTensorType::get({imported->output, imported->batch}, i32);
  auto onnxResultType =
      RankedTensorType::get({imported->batch, imported->output}, i32);

  OwningOpRef<ModuleOp> module = ModuleOp::create(location);
  auto function = func::FuncOp::create(
      location, "main",
      FunctionType::get(&context, onnxInputType, onnxResultType));
  Block *entry = function.addEntryBlock();
  module->push_back(function);
  OpBuilder builder(entry, entry->begin());

  auto weight =
      arith::ConstantOp::create(builder, location, weightType,
                                transposeWeightConstant(builder, *imported));
  auto inputEmpty = tensor::EmptyOp::create(
      builder, location,
      ArrayRef<int64_t>{imported->reduction, imported->batch}, i8);
  auto normalizedInput = linalg::TransposeOp::create(
      builder, location, entry->getArgument(0), inputEmpty.getResult(),
      ArrayRef<int64_t>{1, 0});
  auto zero = arith::ConstantOp::create(
      builder, location, normalizedResultType,
      DenseElementsAttr::get(normalizedResultType,
                             builder.getIntegerAttr(i32, 0)));
  auto matmul = linalg::MatmulOp::create(
      builder, location, TypeRange{normalizedResultType},
      ValueRange{weight.getResult(), normalizedInput.getResult().front()},
      ValueRange{zero.getResult()});
  matmul->setAttr(kMatMulIntegerMarker, builder.getUnitAttr());
  auto outputEmpty = tensor::EmptyOp::create(
      builder, location, ArrayRef<int64_t>{imported->batch, imported->output},
      i32);
  auto output = linalg::TransposeOp::create(
      builder, location, matmul.getResult(0), outputEmpty.getResult(),
      ArrayRef<int64_t>{1, 0});
  func::ReturnOp::create(builder, location, output.getResult().front());
  if (failed(verify(*module)))
    return reject<OwningOpRef<ModuleOp>>(
        context, "constructed MLIR module failed verification");
  return module;
}

static FailureOr<OwningOpRef<ModuleOp>>
importConvInteger(MLIRContext &context, const onnx::ModelProto &model) {
  FailureOr<ConvIntegerModel> imported =
      validateConvIntegerModel(context, model);
  if (failed(imported))
    return failure();

  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<func::FuncDialect>();
  context.getOrLoadDialect<linalg::LinalgDialect>();
  context.getOrLoadDialect<tensor::TensorDialect>();

  Location location = UnknownLoc::get(&context);
  OpBuilder builder(&context);
  auto i8 = builder.getIntegerType(8);
  auto i32 = builder.getIntegerType(32);
  auto inputType = RankedTensorType::get(
      {1, imported->channels, imported->height, imported->width}, i8);
  auto paddedInputType = RankedTensorType::get(
      {1, imported->channels, imported->height + 2, imported->width + 2}, i8);
  auto weightType =
      RankedTensorType::get({imported->filters, imported->channels, 3, 3}, i8);
  auto resultType = RankedTensorType::get(
      {1, imported->filters, imported->height, imported->width}, i32);

  OwningOpRef<ModuleOp> module = ModuleOp::create(location);
  auto function = func::FuncOp::create(
      location, "main", FunctionType::get(&context, inputType, resultType));
  Block *entry = function.addEntryBlock();
  module->push_back(function);
  builder.setInsertionPointToStart(entry);

  auto zeroI8 = arith::ConstantOp::create(builder, location,
                                          builder.getIntegerAttr(i8, 0));
  SmallVector<OpFoldResult> low{
      builder.getIndexAttr(0), builder.getIndexAttr(0), builder.getIndexAttr(1),
      builder.getIndexAttr(1)};
  SmallVector<OpFoldResult> high = low;
  auto paddedInput = tensor::PadOp::create(
      builder, location, paddedInputType, entry->getArgument(0), low, high,
      zeroI8.getResult(), /*nofold=*/false);
  auto weight = arith::ConstantOp::create(
      builder, location, weightType,
      DenseElementsAttr::get(weightType, imported->weightValues));
  auto zeroResult = arith::ConstantOp::create(
      builder, location, resultType,
      DenseElementsAttr::get(resultType, builder.getIntegerAttr(i32, 0)));
  auto vectorType = RankedTensorType::get({2}, builder.getI64Type());
  auto units = DenseIntElementsAttr::get(vectorType, ArrayRef<int64_t>{1, 1});
  auto conv = linalg::Conv2DNchwFchwOp::create(
      builder, location, TypeRange{resultType},
      ValueRange{paddedInput.getResult(), weight.getResult()},
      ValueRange{zeroResult.getResult()}, units, units);
  conv->setAttr(kConvIntegerMarker, builder.getUnitAttr());
  func::ReturnOp::create(builder, location, conv.getResult(0));
  if (failed(verify(*module)))
    return reject<OwningOpRef<ModuleOp>>(
        context, "constructed MLIR module failed verification");
  return module;
}

FailureOr<OwningOpRef<ModuleOp>>
importQuantizedONNX(MLIRContext &context, const onnx::ModelProto &model) {
  const onnx::GraphProto &graph = model.graph();
  if (graph.node_size() != 1 || graph.output_size() != 1)
    return reject<OwningOpRef<ModuleOp>>(
        context, "unsupported output-rooted operator profile");

  const onnx::NodeProto &root = graph.node(0);
  if (root.domain().empty() && root.op_type() == "MatMulInteger")
    return importMatMulInteger(context, model);
  if (root.domain().empty() && root.op_type() == "ConvInteger")
    return importConvInteger(context, model);
  return reject<OwningOpRef<ModuleOp>>(
      context, "unsupported output-rooted operator profile");
}

} // namespace mlir::cim
