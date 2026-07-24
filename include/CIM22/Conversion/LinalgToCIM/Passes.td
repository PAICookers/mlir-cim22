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

def MaterializeCIMInvocation
    : Pass<"materialize-cim-invocation", "::mlir::ModuleOp"> {
  let summary = "Materialize the software-only mapped CIM invocation plan";
  let dependentDialects = ["::mlir::cim::CIMDialect"];
}

#endif // CIM22_CONVERSION_LINALGTOCIM_PASSES
