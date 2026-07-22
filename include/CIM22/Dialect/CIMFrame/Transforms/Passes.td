//===- Passes.td - CIMFrame passes ------------------------*- tablegen -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIM22_DIALECT_CIMFRAME_TRANSFORMS_PASSES
#define CIM22_DIALECT_CIMFRAME_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def LowerCIMFrameCommandsToPackets
    : Pass<"lower-cimframe-commands-to-packets", "::mlir::ModuleOp"> {
  let summary = "Lower CIMFrame commands to typed packets";
  let dependentDialects = ["::mlir::cimframe::CIMFrameDialect"];
}

def VerifyCIMFrame : Pass<"verify-cimframe", "::mlir::ModuleOp"> {
  let summary = "Verify CIMFrame module stage and packet pairing";
}

#endif // CIM22_DIALECT_CIMFRAME_TRANSFORMS_PASSES
