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

def NormalizeCIMConv : Pass<"normalize-cim-conv", "::mlir::ModuleOp"> {
  let summary = "Normalize marked batch-one ConvInteger for CIM lowering";
  let dependentDialects = ["::mlir::arith::ArithDialect",
                           "::mlir::linalg::LinalgDialect",
                           "::mlir::tensor::TensorDialect"];
}

def DumpCIMHostProgram
    : Pass<"dump-cim-host-program", "::mlir::func::FuncOp"> {
  let summary = "Dump the validated read-only Host/CIM program projection";
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
