//===- Passes.td - Linalg to CIM passes -------------------*- tablegen -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIM22_CONVERSION_LINALGTOCIM_PASSES
#define CIM22_CONVERSION_LINALGTOCIM_PASSES

include "mlir/Pass/PassBase.td"

def BindCIMParameters : Pass<"bind-cim-parameters", "::mlir::ModuleOp"> {
  let summary = "Specialize entry arguments using an explicit typed constant manifest";
  let options = [Option<"parameterFile", "file", "std::string", "\"\"",
                        "MLIR module containing cim.parameters function/arg dictionaries">];
  let dependentDialects = ["::mlir::arith::ArithDialect",
                           "::mlir::func::FuncDialect"];
}

def NormalizeCIMConv : Pass<"normalize-cim-conv", "::mlir::ModuleOp"> {
  let summary = "Normalize marked INT8 Conv and static BF16 2D convolution families";
  let description = [{
    BF16 ordinary, grouped and depthwise 2D named convolutions are normalized
    to independent weight-first MatMuls per group. Static batch/spatial axes
    become matrix columns; positive strides and dilations are preserved.
    Weights must be constant and the accumulator must start at positive zero.
    The explicit allow-bias-epilogue opt-in additionally moves FP32 NCHW
    channel-broadcast initializers to a Host epilogue for BF16 preparation.
    Padding remains an explicit tensor.pad on the Host. The marked INT8
    ConvInteger profile retains its batch-one, unit-dilation restrictions.
  }];
  let options = [Option<"allowF32", "allow-f32", "bool", "false",
                        "Also normalize FP32 geometry without changing precision">,
                 Option<"allowBiasEpilogue", "allow-bias-epilogue", "bool", "false",
                        "Opt into FP32 broadcast-init reassociation for BF16 offload">];
  let dependentDialects = ["::mlir::arith::ArithDialect",
                           "::mlir::linalg::LinalgDialect",
                           "::mlir::tensor::TensorDialect"];
}

def NormalizeCIMLinear : Pass<"normalize-cim-linear", "::mlir::ModuleOp"> {
  let summary = "Normalize BF16 activation-first Linear matmul for CIM lowering";
  let options = [Option<"allowF32", "allow-f32", "bool", "false",
                        "Also normalize FP32 Linear without changing precision">];
  let dependentDialects = ["::mlir::arith::ArithDialect",
                           "::mlir::linalg::LinalgDialect",
                           "::mlir::tensor::TensorDialect"];
}

def PrepareCIMBF16 : Pass<"prepare-cim-bf16", "::mlir::ModuleOp"> {
  let summary = "Convert eligible constant-weight FP32 contractions to BF16 with Host casts";
  let dependentDialects = ["::mlir::arith::ArithDialect",
                           "::mlir::linalg::LinalgDialect"];
}

def OutlineCIMBF16 : Pass<"outline-cim-bf16", "::mlir::ModuleOp"> {
  let summary = "Split constant-weight BF16 kernels and Host code using standard Func calls";
  let options = [Option<"maxKernelColumns", "max-kernel-columns", "int64_t", "32",
                        "Stream larger matrices through one-column kernels on Host">];
  let dependentDialects = ["::mlir::arith::ArithDialect",
                           "::mlir::func::FuncDialect",
                           "::mlir::linalg::LinalgDialect",
                           "::mlir::scf::SCFDialect",
                           "::mlir::tensor::TensorDialect"];
}

def FoldCIMInt8Bias : Pass<"fold-cim-int8-bias", "::mlir::ModuleOp"> {
  let summary = "Fold eligible signed INT8 bias into marked CIM MatMul";
  let dependentDialects = ["::mlir::arith::ArithDialect",
                           "::mlir::linalg::LinalgDialect",
                           "::mlir::tensor::TensorDialect"];
  let options = [
    Option<"allowExtraKTile", "allow-extra-k-tile", "bool",
           /*default=*/"true", "Allow exact-K folding to add one K tile">
  ];
}

def VerifyCIMExecutionPlan
    : Pass<"verify-cim-execution-plan", "::mlir::func::FuncOp"> {
  let summary = "Verify the ordered CIM execution plan and Host SSA";
}

def PartitionCIMProgram
    : Pass<"partition-cim-program", "::mlir::ModuleOp"> {
  let summary = "Assign one CIM segment to each offloadable Linalg root";
  let dependentDialects = ["::mlir::linalg::LinalgDialect"];
}

def FormCIMProgram : Pass<"form-cim-program", "::mlir::ModuleOp"> {
  let summary = "Form CIM VMM operations from supported Linalg operations";
  let dependentDialects = ["::mlir::arith::ArithDialect",
                           "::mlir::cim::CIMDialect",
                           "::mlir::linalg::LinalgDialect",
                           "::mlir::tensor::TensorDialect"];
}

def MaterializeCIMSchedule
    : Pass<"materialize-cim-schedule", "::mlir::func::FuncOp"> {
  let summary = "Validate and materialize the software-only CIM schedule";
}

def MaterializeCIMExecutionPlan
    : Pass<"materialize-cim-execution-plan", "::mlir::ModuleOp"> {
  let summary = "Materialize the software-only mapped CIM execution plan";
  let dependentDialects = ["::mlir::cim::CIMDialect"];
}

#endif // CIM22_CONVERSION_LINALGTOCIM_PASSES
