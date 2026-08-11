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
#include "llvm/Support/Endian.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <optional>

namespace mlir::cim {
namespace {
constexpr StringLiteral kMatMulIntegerMarker = "cim.onnx.matmul_integer";
constexpr StringLiteral kConvIntegerMarker = "cim.onnx.conv_integer";

struct MatMulIntegerModel {
  int64_t batch;
  int64_t reduction;
  int64_t output;
  SmallVector<APInt> weightValues;
  SmallVector<APInt> biasValues;
};

struct ConvIntegerModel {
  int64_t channels;
  int64_t inputHeight;
  int64_t inputWidth;
  int64_t filters;
  int64_t kernelHeight;
  int64_t kernelWidth;
  int64_t strideHeight;
  int64_t strideWidth;
  int64_t padTop;
  int64_t padLeft;
  int64_t padBottom;
  int64_t padRight;
  int64_t outputHeight;
  int64_t outputWidth;
  SmallVector<APInt> weightValues;
  SmallVector<APInt> biasValues;
};

struct ConvAttributes {
  std::array<int64_t, 4> pads{0, 0, 0, 0};
  std::array<int64_t, 2> strides{1, 1};
  std::optional<std::array<int64_t, 2>> kernelShape;
};

using InitializerMap = llvm::DenseMap<StringRef, const onnx::TensorProto *>;

struct OutputRoot {
  const onnx::NodeProto *core;
  const onnx::TensorProto *bias;
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
    if (!llvm::isInt<8>(value))
      return reject<SmallVector<APInt>>(
          context, (role + " INT8 int32_data value is out of range").str());
    values.emplace_back(8, static_cast<uint64_t>(value), true);
  }
  return values;
}

FailureOr<SmallVector<APInt>>
extractInt32Initializer(MLIRContext &context, const onnx::TensorProto &tensor,
                        int64_t elementCount, StringRef role) {
  if (tensor.data_type() != onnx::TensorProto::INT32 ||
      tensor.data_location() != onnx::TensorProto::DEFAULT ||
      tensor.external_data_size() != 0)
    return reject<SmallVector<APInt>>(
        context, (role + " must be an embedded INT32 initializer").str());

  SmallVector<APInt> values;
  values.reserve(elementCount);
  if (!tensor.raw_data().empty()) {
    int64_t byteCount = 0;
    if (llvm::MulOverflow(elementCount, int64_t{4}, byteCount) ||
        tensor.raw_data().size() != static_cast<size_t>(byteCount))
      return reject<SmallVector<APInt>>(
          context, (role + " raw_data size does not match its shape").str());
    for (int64_t index = 0; index < elementCount; ++index)
      values.emplace_back(32, llvm::support::endian::read32le(
                                  tensor.raw_data().data() + index * 4));
    return values;
  }

  if (tensor.int32_data_size() != elementCount)
    return reject<SmallVector<APInt>>(
        context,
        (role + " must use raw_data or one int32_data value per element")
            .str());
  for (int32_t value : tensor.int32_data())
    values.emplace_back(32, static_cast<uint32_t>(value));
  return values;
}

FailureOr<OutputRoot> matchOutputRoot(MLIRContext &context,
                                      const onnx::GraphProto &graph,
                                      StringRef coreOpType,
                                      const InitializerMap &initializers) {
  if (graph.output_size() != 1 ||
      (graph.node_size() != 1 && graph.node_size() != 2))
    return reject<OutputRoot>(
        context, "supports one integer core with an optional bias Add");

  const onnx::NodeProto &core = graph.node(0);
  if (!core.domain().empty() || core.op_type() != coreOpType ||
      core.output_size() != 1)
    return reject<OutputRoot>(context,
                              "unsupported output-rooted operator profile");
  if (graph.node_size() == 1) {
    if (core.output(0) != graph.output(0).name())
      return reject<OutputRoot>(context,
                                "integer core output must be the graph output");
    return OutputRoot{&core, nullptr};
  }

  const onnx::NodeProto &add = graph.node(1);
  if (!add.domain().empty() || add.op_type() != "Add" ||
      add.attribute_size() != 0 || add.input_size() != 2 ||
      add.output_size() != 1 || add.input(0) != core.output(0) ||
      add.output(0) != graph.output(0).name())
    return reject<OutputRoot>(
        context, "bias profile requires integer core followed by Add");
  auto bias = initializers.find(add.input(1));
  if (bias == initializers.end())
    return reject<OutputRoot>(context,
                              "bias Add input must be an embedded initializer");
  return OutputRoot{&core, bias->second};
}

FailureOr<ConvAttributes> validateConvAttributes(MLIRContext &context,
                                                 const onnx::NodeProto &node) {
  ConvAttributes result;
  for (const onnx::AttributeProto &attribute : node.attribute()) {
    StringRef name = attribute.name();
    if (name == "pads") {
      if (attribute.type() != onnx::AttributeProto::INTS ||
          attribute.ints_size() != 4)
        return reject<ConvAttributes>(context,
                                      "ConvInteger pads must have four values");
      std::copy(attribute.ints().begin(), attribute.ints().end(),
                result.pads.begin());
      if (!llvm::all_of(result.pads, [](int64_t value) { return value >= 0; }))
        return reject<ConvAttributes>(context,
                                      "ConvInteger pads must be non-negative");
      continue;
    }
    if (name == "strides") {
      if (attribute.type() != onnx::AttributeProto::INTS ||
          attribute.ints_size() != 2)
        return reject<ConvAttributes>(
            context, "ConvInteger strides must have two values");
      std::copy(attribute.ints().begin(), attribute.ints().end(),
                result.strides.begin());
      if (!llvm::all_of(result.strides,
                        [](int64_t value) { return value > 0; }))
        return reject<ConvAttributes>(context,
                                      "ConvInteger strides must be positive");
      continue;
    }
    if (name == "dilations" && attribute.type() == onnx::AttributeProto::INTS &&
        attribute.ints_size() == 2 &&
        llvm::all_of(attribute.ints(),
                     [](int64_t value) { return value == 1; }))
      continue;
    if (name == "kernel_shape") {
      if (attribute.type() != onnx::AttributeProto::INTS ||
          attribute.ints_size() != 2 ||
          !llvm::all_of(attribute.ints(),
                        [](int64_t value) { return value > 0; }))
        return reject<ConvAttributes>(
            context, "ConvInteger kernel_shape must have two positive values");
      result.kernelShape =
          std::array<int64_t, 2>{attribute.ints(0), attribute.ints(1)};
      continue;
    }
    if (name == "group" && attribute.type() == onnx::AttributeProto::INT &&
        attribute.i() == 1)
      continue;
    return reject<ConvAttributes>(context,
                                  "ConvInteger has an unsupported attribute");
  }
  return result;
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
  InitializerMap initializers;
  for (const onnx::TensorProto &initializer : graph.initializer()) {
    if (initializer.name().empty() ||
        !initializers.try_emplace(initializer.name(), &initializer).second)
      return reject<MatMulIntegerModel>(
          context, "initializer names must be non-empty and unique");
  }
  FailureOr<OutputRoot> root =
      matchOutputRoot(context, graph, "MatMulInteger", initializers);
  if (failed(root))
    return failure();

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

  const onnx::NodeProto &node = *root->core;
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
  int64_t weightElementCount = 0;
  if (llvm::MulOverflow(reduction, output, weightElementCount))
    return reject<MatMulIntegerModel>(context,
                                      "weight element count is too large");

  FailureOr<SmallVector<APInt>> weightValues =
      extractInt8Initializer(context, weight, weightElementCount);
  if (failed(weightValues))
    return failure();
  SmallVector<APInt> biasValues;
  if (root->bias) {
    if (root->bias->dims_size() != 1 || root->bias->dims(0) != output)
      return reject<MatMulIntegerModel>(
          context, "MatMulInteger bias must have shape [N]");
    FailureOr<SmallVector<APInt>> values =
        extractInt32Initializer(context, *root->bias, output, "bias");
    if (failed(values))
      return failure();
    biasValues = std::move(*values);
  }
  return MatMulIntegerModel{batch, reduction, output, std::move(*weightValues),
                            std::move(biasValues)};
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
  InitializerMap initializers;
  for (const onnx::TensorProto &initializer : graph.initializer()) {
    if (initializer.name().empty() ||
        !initializers.try_emplace(initializer.name(), &initializer).second)
      return reject<ConvIntegerModel>(
          context, "initializer names must be non-empty and unique");
  }
  FailureOr<OutputRoot> root =
      matchOutputRoot(context, graph, "ConvInteger", initializers);
  if (failed(root))
    return failure();

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

  const onnx::NodeProto &node = *root->core;
  if (!node.domain().empty() || node.op_type() != "ConvInteger" ||
      node.input_size() < 2 || node.input_size() > 4 || node.output_size() != 1)
    return reject<ConvIntegerModel>(
        context, "supports only the frozen ConvInteger profile");
  FailureOr<ConvAttributes> attributes = validateConvAttributes(context, node);
  if (failed(attributes))
    return failure();
  if (node.input(0) != runtimeInputs.front()->name())
    return reject<ConvIntegerModel>(
        context, "ConvInteger input must be the runtime graph input");
  auto weightIt = initializers.find(node.input(1));
  if (weightIt == initializers.end())
    return reject<ConvIntegerModel>(
        context, "ConvInteger weight must be an embedded initializer");
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
      weight.dims(2) <= 0 || weight.dims(3) <= 0)
    return reject<ConvIntegerModel>(
        context, "ConvInteger weight must have four positive dimensions");
  int64_t channels = (*inputShape)[1];
  int64_t inputHeight = (*inputShape)[2];
  int64_t inputWidth = (*inputShape)[3];
  int64_t filters = weight.dims(0);
  int64_t kernelHeight = weight.dims(2);
  int64_t kernelWidth = weight.dims(3);
  if (attributes->kernelShape &&
      *attributes->kernelShape !=
          std::array<int64_t, 2>{kernelHeight, kernelWidth})
    return reject<ConvIntegerModel>(
        context, "ConvInteger kernel_shape must match the weight shape");
  int64_t paddedHeight = 0;
  int64_t paddedWidth = 0;
  int64_t partialHeight = 0;
  int64_t partialWidth = 0;
  int64_t outputHeight = 0;
  int64_t outputWidth = 0;
  int64_t kernelElements = 0;
  int64_t weightElementCount = 0;
  if (llvm::AddOverflow(inputHeight, attributes->pads[0], partialHeight) ||
      llvm::AddOverflow(partialHeight, attributes->pads[2], paddedHeight) ||
      llvm::AddOverflow(inputWidth, attributes->pads[1], partialWidth) ||
      llvm::AddOverflow(partialWidth, attributes->pads[3], paddedWidth) ||
      paddedHeight < kernelHeight || paddedWidth < kernelWidth ||
      llvm::AddOverflow((paddedHeight - kernelHeight) / attributes->strides[0],
                        int64_t{1}, outputHeight) ||
      llvm::AddOverflow((paddedWidth - kernelWidth) / attributes->strides[1],
                        int64_t{1}, outputWidth) ||
      llvm::MulOverflow(channels, kernelHeight, kernelElements) ||
      llvm::MulOverflow(kernelElements, kernelWidth, kernelElements) ||
      llvm::MulOverflow(filters, kernelElements, weightElementCount))
    return reject<ConvIntegerModel>(context, "ConvInteger shape is too large");
  if (weight.dims(1) != channels || (*outputShape)[1] != filters ||
      (*outputShape)[2] != outputHeight || (*outputShape)[3] != outputWidth)
    return reject<ConvIntegerModel>(
        context,
        "ConvInteger input, weight, and output shapes are inconsistent");

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
      extractInt8Initializer(context, weight, weightElementCount, "weight");
  if (failed(weightValues))
    return failure();
  SmallVector<APInt> biasValues;
  if (root->bias) {
    if (root->bias->dims_size() != 4 || root->bias->dims(0) != 1 ||
        root->bias->dims(1) != filters || root->bias->dims(2) != 1 ||
        root->bias->dims(3) != 1)
      return reject<ConvIntegerModel>(
          context, "ConvInteger bias must have shape [1,F,1,1]");
    FailureOr<SmallVector<APInt>> values =
        extractInt32Initializer(context, *root->bias, filters, "bias");
    if (failed(values))
      return failure();
    biasValues = std::move(*values);
  }
  return ConvIntegerModel{channels,
                          inputHeight,
                          inputWidth,
                          filters,
                          kernelHeight,
                          kernelWidth,
                          attributes->strides[0],
                          attributes->strides[1],
                          attributes->pads[0],
                          attributes->pads[1],
                          attributes->pads[2],
                          attributes->pads[3],
                          outputHeight,
                          outputWidth,
                          std::move(*weightValues),
                          std::move(biasValues)};
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

Value appendIntegerBias(OpBuilder &builder, Location location, Value output,
                        RankedTensorType resultType,
                        ArrayRef<int64_t> biasShape, ArrayRef<APInt> biasValues,
                        ArrayRef<int64_t> broadcastDimensions) {
  if (biasValues.empty())
    return output;

  auto biasType = RankedTensorType::get(biasShape, resultType.getElementType());
  auto bias =
      arith::ConstantOp::create(builder, location, biasType,
                                DenseElementsAttr::get(biasType, biasValues));
  auto empty = tensor::EmptyOp::create(builder, location, resultType.getShape(),
                                       resultType.getElementType());
  auto broadcast =
      linalg::BroadcastOp::create(builder, location, bias.getResult(),
                                  empty.getResult(), broadcastDimensions);
  return linalg::AddOp::create(
             builder, location, TypeRange{resultType},
             ValueRange{output, broadcast.getResult().front()},
             ValueRange{empty.getResult()})
      .getResultTensors()
      .front();
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
  Value result =
      appendIntegerBias(builder, location, output.getResult().front(),
                        onnxResultType, ArrayRef<int64_t>{imported->output},
                        imported->biasValues, ArrayRef<int64_t>{0});
  func::ReturnOp::create(builder, location, result);
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
      {1, imported->channels, imported->inputHeight, imported->inputWidth}, i8);
  auto weightType =
      RankedTensorType::get({imported->filters, imported->channels,
                             imported->kernelHeight, imported->kernelWidth},
                            i8);
  auto resultType = RankedTensorType::get(
      {1, imported->filters, imported->outputHeight, imported->outputWidth},
      i32);

  OwningOpRef<ModuleOp> module = ModuleOp::create(location);
  auto function = func::FuncOp::create(
      location, "main", FunctionType::get(&context, inputType, resultType));
  Block *entry = function.addEntryBlock();
  module->push_back(function);
  builder.setInsertionPointToStart(entry);

  Value paddedInput = entry->getArgument(0);
  if (imported->padTop != 0 || imported->padLeft != 0 ||
      imported->padBottom != 0 || imported->padRight != 0) {
    auto paddedInputType = RankedTensorType::get(
        {1, imported->channels,
         imported->inputHeight + imported->padTop + imported->padBottom,
         imported->inputWidth + imported->padLeft + imported->padRight},
        i8);
    auto zeroI8 = arith::ConstantOp::create(builder, location,
                                            builder.getIntegerAttr(i8, 0));
    SmallVector<OpFoldResult> low{builder.getIndexAttr(0),
                                  builder.getIndexAttr(0),
                                  builder.getIndexAttr(imported->padTop),
                                  builder.getIndexAttr(imported->padLeft)};
    SmallVector<OpFoldResult> high{builder.getIndexAttr(0),
                                   builder.getIndexAttr(0),
                                   builder.getIndexAttr(imported->padBottom),
                                   builder.getIndexAttr(imported->padRight)};
    paddedInput = tensor::PadOp::create(builder, location, paddedInputType,
                                        entry->getArgument(0), low, high,
                                        zeroI8.getResult(), /*nofold=*/false);
  }
  auto weight = arith::ConstantOp::create(
      builder, location, weightType,
      DenseElementsAttr::get(weightType, imported->weightValues));
  auto zeroResult = arith::ConstantOp::create(
      builder, location, resultType,
      DenseElementsAttr::get(resultType, builder.getIntegerAttr(i32, 0)));
  auto vectorType = RankedTensorType::get({2}, builder.getI64Type());
  auto strides = DenseIntElementsAttr::get(
      vectorType,
      ArrayRef<int64_t>{imported->strideHeight, imported->strideWidth});
  auto dilations =
      DenseIntElementsAttr::get(vectorType, ArrayRef<int64_t>{1, 1});
  auto conv = linalg::Conv2DNchwFchwOp::create(
      builder, location, TypeRange{resultType},
      ValueRange{paddedInput, weight.getResult()},
      ValueRange{zeroResult.getResult()}, strides, dilations);
  conv->setAttr(kConvIntegerMarker, builder.getUnitAttr());
  Value result =
      appendIntegerBias(builder, location, conv.getResult(0), resultType,
                        ArrayRef<int64_t>{imported->filters},
                        imported->biasValues, ArrayRef<int64_t>{0, 2, 3});
  func::ReturnOp::create(builder, location, result);
  if (failed(verify(*module)))
    return reject<OwningOpRef<ModuleOp>>(
        context, "constructed MLIR module failed verification");
  return module;
}

FailureOr<OwningOpRef<ModuleOp>>
importQuantizedONNX(MLIRContext &context, const onnx::ModelProto &model) {
  const onnx::GraphProto &graph = model.graph();
  if ((graph.node_size() != 1 && graph.node_size() != 2) ||
      graph.output_size() != 1)
    return reject<OwningOpRef<ModuleOp>>(
        context, "unsupported output-rooted operator profile");

  const onnx::NodeProto &core = graph.node(0);
  if (core.domain().empty() && core.op_type() == "MatMulInteger")
    return importMatMulInteger(context, model);
  if (core.domain().empty() && core.op_type() == "ConvInteger")
    return importConvInteger(context, model);
  return reject<OwningOpRef<ModuleOp>>(
      context, "unsupported output-rooted operator profile");
}

} // namespace mlir::cim
