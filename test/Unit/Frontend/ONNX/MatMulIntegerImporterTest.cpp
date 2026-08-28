//===- MatMulIntegerImporterTest.cpp - ONNX importer regression -*- C++ -*-===//
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

constexpr llvm::StringLiteral KMatMulIntegerMarker = "cim.onnx.matmul_integer";

static bool check(bool Condition, llvm::StringRef Message) {
  if (Condition)
    return true;
  llvm::errs() << "FAIL: " << Message << '\n';
  return false;
}

static bool isTensorType(mlir::Type Type, llvm::ArrayRef<int64_t> Shape,
                         unsigned Width) {
  auto Tensor = mlir::dyn_cast<mlir::RankedTensorType>(Type);
  auto Element =
      Tensor ? mlir::dyn_cast<mlir::IntegerType>(Tensor.getElementType())
             : mlir::IntegerType{};
  return Tensor && Tensor.getShape() == Shape && Element &&
         Element.getWidth() == Width;
}

static bool rejects(mlir::MLIRContext &Context, const onnx::ModelProto &Model) {
  mlir::ScopedDiagnosticHandler Silence(
      &Context, [](mlir::Diagnostic &) { return mlir::success(); });
  return mlir::failed(mlir::cim::importQuantizedONNX(Context, Model));
}

static bool checkProfileRejections(mlir::MLIRContext &Context,
                                   const onnx::ModelProto &Model) {
  auto UnknownRoot = Model;
  UnknownRoot.mutable_graph()->mutable_node(0)->set_op_type("UnknownRoot");

  auto WrongOpset = Model;
  WrongOpset.mutable_opset_import(0)->set_version(11);

  auto CustomDomain = Model;
  CustomDomain.mutable_graph()->mutable_node(0)->set_domain("example");

  auto WrongActivationType = Model;
  WrongActivationType.mutable_graph()
      ->mutable_input(0)
      ->mutable_type()
      ->mutable_tensor_type()
      ->set_elem_type(onnx::TensorProto::UINT8);

  auto WrongOutputType = Model;
  WrongOutputType.mutable_graph()
      ->mutable_output(0)
      ->mutable_type()
      ->mutable_tensor_type()
      ->set_elem_type(onnx::TensorProto::INT64);

  auto ZeroDimension = Model;
  ZeroDimension.mutable_graph()
      ->mutable_input(0)
      ->mutable_type()
      ->mutable_tensor_type()
      ->mutable_shape()
      ->mutable_dim(1)
      ->set_dim_value(0);

  auto DynamicDimension = Model;
  auto *DynamicDim = DynamicDimension.mutable_graph()
                         ->mutable_input(0)
                         ->mutable_type()
                         ->mutable_tensor_type()
                         ->mutable_shape()
                         ->mutable_dim(1);
  DynamicDim->clear_dim_value();
  DynamicDim->set_dim_param("K");

  auto IncompatibleShape = Model;
  IncompatibleShape.mutable_graph()
      ->mutable_output(0)
      ->mutable_type()
      ->mutable_tensor_type()
      ->mutable_shape()
      ->mutable_dim(1)
      ->set_dim_value(1023);

  auto RuntimeWeight = Model;
  RuntimeWeight.mutable_graph()->clear_initializer();

  auto ExplicitZeroPoint = Model;
  ExplicitZeroPoint.mutable_graph()->mutable_node(0)->add_input("a_zero_point");

  auto ExternalWeight = Model;
  ExternalWeight.mutable_graph()->mutable_initializer(0)->set_data_location(
      onnx::TensorProto::EXTERNAL);

  auto UnsupportedOperation = Model;
  UnsupportedOperation.mutable_graph()->add_node()->set_op_type("Relu");

  return check(rejects(Context, UnknownRoot),
               "an unknown graph root must be rejected") &&
         check(rejects(Context, WrongOpset),
               "wrong default opset must reject") &&
         check(rejects(Context, CustomDomain),
               "custom core domain must reject") &&
         check(rejects(Context, WrongActivationType),
               "non-INT8 activation must reject") &&
         check(rejects(Context, WrongOutputType),
               "non-INT32 result must reject") &&
         check(rejects(Context, ZeroDimension), "zero dimension must reject") &&
         check(rejects(Context, DynamicDimension),
               "dynamic dimension must reject") &&
         check(rejects(Context, IncompatibleShape),
               "incompatible shape must reject") &&
         check(rejects(Context, RuntimeWeight), "runtime weight must reject") &&
         check(rejects(Context, ExplicitZeroPoint),
               "explicit zero-point must reject") &&
         check(rejects(Context, ExternalWeight),
               "external weight must reject") &&
         check(rejects(Context, UnsupportedOperation),
               "unsupported additional operation must reject");
}

static int32_t getWeightValue(const onnx::TensorProto &Weight, int64_t Index) {
  if (!Weight.raw_data().empty())
    return static_cast<int8_t>(Weight.raw_data()[Index]);
  return Weight.int32_data(static_cast<int>(Index));
}

static bool hasTransposedWeight(mlir::Value Value,
                                const onnx::TensorProto &OnnxWeight) {
  auto Constant = Value.getDefiningOp<mlir::arith::ConstantOp>();
  auto Elements =
      Constant ? mlir::dyn_cast<mlir::DenseElementsAttr>(Constant.getValue())
               : mlir::DenseElementsAttr{};
  if (!Elements || OnnxWeight.dims_size() != 2)
    return false;

  int64_t Reduction = OnnxWeight.dims(0);
  int64_t Output = OnnxWeight.dims(1);
  if (Elements.getNumElements() != Reduction * Output)
    return false;
  int64_t Index = 0;
  for (llvm::APInt Actual : Elements.getValues<llvm::APInt>()) {
    int64_t OutputIndex = Index / Reduction;
    int64_t ReductionIndex = Index % Reduction;
    int32_t Expected =
        getWeightValue(OnnxWeight, ReductionIndex * Output + OutputIndex);
    if (Actual.getSExtValue() != Expected)
      return false;
    ++Index;
  }
  return true;
}

static bool checkNormalizedModule(mlir::ModuleOp Module,
                                  const onnx::TensorProto &OnnxWeight,
                                  int64_t Batch, int64_t Reduction,
                                  int64_t Output) {
  auto Function = Module.lookupSymbol<mlir::func::FuncOp>("main");
  if (!check(Function && Function.getFunctionType().getNumInputs() == 1 &&
                 Function.getFunctionType().getNumResults() == 1 &&
                 isTensorType(Function.getFunctionType().getInput(0),
                              {Batch, Reduction}, 8) &&
                 isTensorType(Function.getFunctionType().getResult(0),
                              {Batch, Output}, 32),
             "ONNX input and result contract must be preserved"))
    return false;

  unsigned MatmulCount = 0;
  unsigned TransposeCount = 0;
  bool NormalizedMatmul = false;
  bool TransposedWeight = false;
  Module.walk([&](mlir::linalg::MatmulOp Matmul) {
    ++MatmulCount;
    auto Inputs = Matmul.getDpsInputs();
    NormalizedMatmul =
        Inputs.size() == 2 &&
        isTensorType(Inputs[0].getType(), {Output, Reduction}, 8) &&
        isTensorType(Inputs[1].getType(), {Reduction, Batch}, 8) &&
        isTensorType(Matmul.getResult(0).getType(), {Output, Batch}, 32) &&
        llvm::isa<mlir::UnitAttr>(Matmul->getAttr(KMatMulIntegerMarker));
    TransposedWeight = hasTransposedWeight(Inputs[0], OnnxWeight);
  });
  Module.walk([&](mlir::linalg::TransposeOp) { ++TransposeCount; });
  return check(MatmulCount == 1 && NormalizedMatmul && TransposedWeight &&
                   TransposeCount == 2,
               "importer must materialize the weight-first axis normalization");
}

static bool checkLowering(mlir::ModuleOp Module, int64_t Batch,
                          int64_t Reduction, int64_t Output) {
  mlir::PassManager PassManager(Module.getContext());
  PassManager.addPass(mlir::cim::createPartitionCIMProgram());
  PassManager.addPass(mlir::cim::createFormCIMProgram());
  if (mlir::failed(PassManager.run(Module)))
    return check(false,
                 "M1.8 pass must lower the imported MatMulInteger module");

  unsigned VmmCount = 0;
  unsigned ExtensionCount = 0;
  unsigned AddCount = 0;
  unsigned TruncateCount = 0;
  unsigned ResidualContractions = 0;
  Module.walk([&](mlir::Operation *Operation) {
    VmmCount += llvm::isa<mlir::cim::VMMOp>(Operation);
    ExtensionCount += llvm::isa<mlir::arith::ExtSIOp>(Operation);
    AddCount += llvm::isa<mlir::arith::AddIOp>(Operation);
    TruncateCount += llvm::isa<mlir::arith::TruncIOp>(Operation);
    ResidualContractions +=
        llvm::isa<mlir::linalg::MatvecOp, mlir::linalg::MatmulOp>(Operation);
  });
  uint64_t OutputTiles = (Output + 15) / 16;
  uint64_t ReductionTiles = (Reduction + 63) / 64;
  uint64_t Partials = Batch * OutputTiles * ReductionTiles;
  uint64_t Adds = Batch * OutputTiles * (ReductionTiles - 1);
  return check(VmmCount == Partials && ExtensionCount == Partials &&
                   AddCount == Adds && TruncateCount == 0 &&
                   ResidualContractions == 0,
               "M1.8 must retain i21 partials and only i32 Host reduction");
}

static onnx::ModelProto makeSyntheticModel(const onnx::ModelProto &Fixture) {
  onnx::ModelProto Model = Fixture;
  constexpr int64_t Batch = 2;
  constexpr int64_t Reduction = 65;
  constexpr int64_t Output = 17;
  Model.set_ir_version(9);
  auto *UnusedOpset = Model.add_opset_import();
  UnusedOpset->set_domain("example.unused");
  UnusedOpset->set_version(1);
  auto *Graph = Model.mutable_graph();
  auto *InputShape = Graph->mutable_input(0)
                         ->mutable_type()
                         ->mutable_tensor_type()
                         ->mutable_shape();
  InputShape->mutable_dim(0)->set_dim_value(Batch);
  InputShape->mutable_dim(1)->set_dim_value(Reduction);
  auto *OutputShape = Graph->mutable_output(0)
                          ->mutable_type()
                          ->mutable_tensor_type()
                          ->mutable_shape();
  OutputShape->mutable_dim(0)->set_dim_value(Batch);
  OutputShape->mutable_dim(1)->set_dim_value(Output);
  auto *Weight = Graph->mutable_initializer(0);
  Weight->set_dims(0, Reduction);
  Weight->set_dims(1, Output);
  Weight->clear_raw_data();
  Weight->clear_int32_data();
  for (int64_t Index = 0; Index < Reduction * Output; ++Index)
    Weight->add_int32_data(static_cast<int32_t>(Index % 7) - 3);
  return Model;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    llvm::errs()
        << "usage: mlir-cim22-matmulinteger-importer-test <model.onnx>\n";
    return 1;
  }

  onnx::ModelProto Model;
  std::ifstream Input(argv[1], std::ios::binary);
  if (!check(Input && Model.ParseFromIstream(&Input), "fixture must parse"))
    return 1;

  mlir::MLIRContext Context;
  Context.loadDialect<mlir::arith::ArithDialect, mlir::cim::CIMDialect,
                      mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                      mlir::tensor::TensorDialect>();
  if (!checkProfileRejections(Context, Model))
    return 1;

  auto Module = mlir::cim::importQuantizedONNX(Context, Model);
  if (!check(mlir::succeeded(Module), "fixture must import"))
    return 1;
  mlir::ModuleOp Imported = (*Module).get();
  if (!checkNormalizedModule(Imported, Model.graph().initializer(0), 32, 512,
                             1024) ||
      !checkLowering(Imported, 32, 512, 1024))
    return 1;

  onnx::ModelProto Synthetic = makeSyntheticModel(Model);
  auto SyntheticModule = mlir::cim::importQuantizedONNX(Context, Synthetic);
  if (!check(mlir::succeeded(SyntheticModule),
             "non-fixture static MatMulInteger must import"))
    return 1;
  mlir::ModuleOp SyntheticImported = (*SyntheticModule).get();
  if (!checkNormalizedModule(SyntheticImported,
                             Synthetic.graph().initializer(0), 2, 65, 17) ||
      !checkLowering(SyntheticImported, 2, 65, 17))
    return 1;

  llvm::outs() << "PASS\n";
  return 0;
}
