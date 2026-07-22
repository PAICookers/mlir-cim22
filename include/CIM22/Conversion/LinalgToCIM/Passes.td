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
  let dependentDialects = ["::mlir::cim::CIMDialect",
                           "::mlir::tensor::TensorDialect"];
}

#endif // CIM22_CONVERSION_LINALGTOCIM_PASSES
