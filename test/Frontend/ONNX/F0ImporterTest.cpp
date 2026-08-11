//===- F0ImporterTest.cpp - Quantized ONNX importer regression -*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIM22/Conversion/LinalgToCIM/Passes.h"
#include "CIM22/Dialect/CIM/IR/CIMDialect.h"
#include "CIM22/Frontend/ONNX/Importer.h"

#include "onnx/onnx_pb.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <fstream>

namespace {
constexpr llvm::StringLiteral kMatMulIntegerMarker = "cim.onnx.matmul_integer";

bool check(bool condition, llvm::StringRef message) {
  if (condition)
    return true;
  llvm::errs() << "FAIL: " << message << '\n';
  return false;
}

bool isTensorType(mlir::Type type, llvm::ArrayRef<int64_t> shape,
                  unsigned width) {
  auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(type);
  auto element =
      tensor ? mlir::dyn_cast<mlir::IntegerType>(tensor.getElementType())
             : mlir::IntegerType{};
  return tensor && tensor.getShape() == shape && element &&
         element.getWidth() == width;
}

bool rejects(mlir::MLIRContext &context, const onnx::ModelProto &model) {
  mlir::ScopedDiagnosticHandler silence(
      &context, [](mlir::Diagnostic &) { return mlir::success(); });
  return mlir::failed(mlir::cim::importQuantizedONNX(context, model));
}

bool checkUnknownRootRejection(mlir::MLIRContext &context,
                               const onnx::ModelProto &model) {
  auto unknownRoot = model;
  unknownRoot.mutable_graph()->mutable_node(0)->set_op_type("UnknownRoot");
  return check(rejects(context, unknownRoot),
               "an unknown graph root must be rejected");
}

int32_t getWeightValue(const onnx::TensorProto &weight, int64_t index) {
  if (!weight.raw_data().empty())
    return static_cast<int8_t>(weight.raw_data()[index]);
  return weight.int32_data(static_cast<int>(index));
}

bool hasTransposedWeight(mlir::Value value,
                         const onnx::TensorProto &onnxWeight) {
  auto constant = value.getDefiningOp<mlir::arith::ConstantOp>();
  auto elements =
      constant ? mlir::dyn_cast<mlir::DenseElementsAttr>(constant.getValue())
               : mlir::DenseElementsAttr{};
  if (!elements || onnxWeight.dims_size() != 2)
    return false;

  int64_t reduction = onnxWeight.dims(0);
  int64_t output = onnxWeight.dims(1);
  if (elements.getNumElements() != reduction * output)
    return false;
  int64_t index = 0;
  for (llvm::APInt actual : elements.getValues<llvm::APInt>()) {
    int64_t outputIndex = index / reduction;
    int64_t reductionIndex = index % reduction;
    int32_t expected =
        getWeightValue(onnxWeight, reductionIndex * output + outputIndex);
    if (actual.getSExtValue() != expected)
      return false;
    ++index;
  }
  return true;
}

bool checkNormalizedModule(mlir::ModuleOp module,
                           const onnx::TensorProto &onnxWeight, int64_t batch,
                           int64_t reduction, int64_t output) {
  auto function = module.lookupSymbol<mlir::func::FuncOp>("main");
  if (!check(function && function.getFunctionType().getNumInputs() == 1 &&
                 function.getFunctionType().getNumResults() == 1 &&
                 isTensorType(function.getFunctionType().getInput(0),
                              {batch, reduction}, 8) &&
                 isTensorType(function.getFunctionType().getResult(0),
                              {batch, output}, 32),
             "ONNX input and result contract must be preserved"))
    return false;

  unsigned matmulCount = 0;
  unsigned transposeCount = 0;
  bool normalizedMatmul = false;
  bool transposedWeight = false;
  module.walk([&](mlir::linalg::MatmulOp matmul) {
    ++matmulCount;
    auto inputs = matmul.getDpsInputs();
    normalizedMatmul =
        inputs.size() == 2 &&
        isTensorType(inputs[0].getType(), {output, reduction}, 8) &&
        isTensorType(inputs[1].getType(), {reduction, batch}, 8) &&
        isTensorType(matmul.getResult(0).getType(), {output, batch}, 32) &&
        llvm::isa<mlir::UnitAttr>(matmul->getAttr(kMatMulIntegerMarker));
    transposedWeight = hasTransposedWeight(inputs[0], onnxWeight);
  });
  module.walk([&](mlir::linalg::TransposeOp) { ++transposeCount; });
  return check(matmulCount == 1 && normalizedMatmul && transposedWeight &&
                   transposeCount == 2,
               "importer must materialize the weight-first axis normalization");
}

bool checkLowering(mlir::ModuleOp module, int64_t batch, int64_t reduction,
                   int64_t output) {
  mlir::PassManager passManager(module.getContext());
  passManager.addPass(mlir::cim::createFormCIMProgram());
  if (mlir::failed(passManager.run(module)))
    return check(false,
                 "M1.8 pass must lower the imported MatMulInteger module");

  unsigned vmmCount = 0;
  unsigned extensionCount = 0;
  unsigned addCount = 0;
  unsigned truncateCount = 0;
  unsigned residualContractions = 0;
  module.walk([&](mlir::Operation *operation) {
    vmmCount += llvm::isa<mlir::cim::VMMOp>(operation);
    extensionCount += llvm::isa<mlir::arith::ExtSIOp>(operation);
    addCount += llvm::isa<mlir::arith::AddIOp>(operation);
    truncateCount += llvm::isa<mlir::arith::TruncIOp>(operation);
    residualContractions +=
        llvm::isa<mlir::linalg::MatvecOp, mlir::linalg::MatmulOp>(operation);
  });
  uint64_t outputTiles = (output + 15) / 16;
  uint64_t reductionTiles = (reduction + 63) / 64;
  uint64_t partials = batch * outputTiles * reductionTiles;
  uint64_t adds = batch * outputTiles * (reductionTiles - 1);
  return check(vmmCount == partials && extensionCount == partials &&
                   addCount == adds && truncateCount == 0 &&
                   residualContractions == 0,
               "M1.8 must retain i21 partials and only i32 Host reduction");
}

onnx::ModelProto makeSyntheticModel(const onnx::ModelProto &fixture) {
  onnx::ModelProto model = fixture;
  constexpr int64_t batch = 2;
  constexpr int64_t reduction = 65;
  constexpr int64_t output = 17;
  model.set_ir_version(9);
  auto *unusedOpset = model.add_opset_import();
  unusedOpset->set_domain("example.unused");
  unusedOpset->set_version(1);
  auto *graph = model.mutable_graph();
  auto *inputShape = graph->mutable_input(0)
                         ->mutable_type()
                         ->mutable_tensor_type()
                         ->mutable_shape();
  inputShape->mutable_dim(0)->set_dim_value(batch);
  inputShape->mutable_dim(1)->set_dim_value(reduction);
  auto *outputShape = graph->mutable_output(0)
                          ->mutable_type()
                          ->mutable_tensor_type()
                          ->mutable_shape();
  outputShape->mutable_dim(0)->set_dim_value(batch);
  outputShape->mutable_dim(1)->set_dim_value(output);
  auto *weight = graph->mutable_initializer(0);
  weight->set_dims(0, reduction);
  weight->set_dims(1, output);
  weight->clear_raw_data();
  weight->clear_int32_data();
  for (int64_t index = 0; index < reduction * output; ++index)
    weight->add_int32_data(static_cast<int32_t>(index % 7) - 3);
  return model;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    llvm::errs() << "usage: mlir-cim22-f0-importer-test <model.onnx>\n";
    return 1;
  }

  onnx::ModelProto model;
  std::ifstream input(argv[1], std::ios::binary);
  if (!check(input && model.ParseFromIstream(&input), "fixture must parse"))
    return 1;

  mlir::MLIRContext context;
  context.loadDialect<mlir::arith::ArithDialect, mlir::cim::CIMDialect,
                      mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                      mlir::tensor::TensorDialect>();
  if (!checkUnknownRootRejection(context, model))
    return 1;

  auto module = mlir::cim::importQuantizedONNX(context, model);
  if (!check(mlir::succeeded(module), "fixture must import"))
    return 1;
  mlir::ModuleOp imported = (*module).get();
  if (!checkNormalizedModule(imported, model.graph().initializer(0), 32, 512,
                             1024) ||
      !checkLowering(imported, 32, 512, 1024))
    return 1;

  onnx::ModelProto synthetic = makeSyntheticModel(model);
  auto syntheticModule = mlir::cim::importQuantizedONNX(context, synthetic);
  if (!check(mlir::succeeded(syntheticModule),
             "non-fixture static MatMulInteger must import"))
    return 1;
  mlir::ModuleOp syntheticImported = (*syntheticModule).get();
  if (!checkNormalizedModule(syntheticImported,
                             synthetic.graph().initializer(0), 2, 65, 17) ||
      !checkLowering(syntheticImported, 2, 65, 17))
    return 1;

  llvm::outs() << "PASS\n";
  return 0;
}
