//===- Passes.td - CIM22 target passes -------------------*- tablegen -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIM22_TARGET_PASSES
#define CIM22_TARGET_PASSES

include "mlir/Pass/PassBase.td"

def MapCIMSchedule : Pass<"map-cim-schedule", "::mlir::func::FuncOp"> {
  let summary = "Map a logical CIM schedule to the selected target profile";
}

#endif // CIM22_TARGET_PASSES
