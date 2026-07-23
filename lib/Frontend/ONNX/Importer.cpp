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

struct MatMulIntegerModel {
  int64_t batch;
  int64_t reduction;
  int64_t output;
  SmallVector<APInt> weightValues;
};

template <typename T>
FailureOr<T> reject(MLIRContext &context, StringRef reason) {
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
                       int64_t elementCount) {
  if (tensor.data_type() != onnx::TensorProto::INT8 ||
      tensor.data_location() != onnx::TensorProto::DEFAULT ||
      tensor.external_data_size() != 0)
    return reject<SmallVector<APInt>>(
        context, "weight must be an embedded INT8 initializer");

  SmallVector<APInt> values;
  values.reserve(elementCount);
  if (!tensor.raw_data().empty()) {
    if (tensor.raw_data().size() != static_cast<size_t>(elementCount))
      return reject<SmallVector<APInt>>(
          context, "weight raw_data size does not match its shape");
    for (unsigned char byte : tensor.raw_data())
      values.emplace_back(8, byte);
    return values;
  }

  if (tensor.int32_data_size() != elementCount)
    return reject<SmallVector<APInt>>(
        context,
        "weight must use raw_data or one int32_data value per element");
  for (int32_t value : tensor.int32_data()) {
    if (value < -128 || value > 127)
      return reject<SmallVector<APInt>>(
          context, "INT8 weight int32_data value is out of range");
    values.emplace_back(8, static_cast<uint64_t>(value), true);
  }
  return values;
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

FailureOr<OwningOpRef<ModuleOp>>
importQuantizedONNX(MLIRContext &context, const onnx::ModelProto &model) {
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

} // namespace mlir::cim
