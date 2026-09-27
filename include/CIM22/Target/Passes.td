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

def ExportCIMRTLArtifacts
    : Pass<"export-cim-rtl-artifacts", "::mlir::ModuleOp"> {
  let summary = "Export an INT8 RTL transaction or explicit BF16 reference case";
  let options = [
    Option<"outputDir", "output-dir", "std::string", /*default=*/"",
           "New directory that receives the transaction artifacts">,
    Option<"inputCacheFile", "input-cache-file", "std::string",
           /*default=*/"", "TB-compatible 8-row input cache for dynamic inputs">,
    Option<"bf16Reference", "bf16-reference", "bool", /*default=*/"false",
           "Export provisional supplier-model BF16 reference artifacts for comparison">
  ];
}

#endif // CIM22_TARGET_PASSES
