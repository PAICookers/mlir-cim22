//===- CIMDialect.td - CIM dialect -----------------------*- tablegen -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIM22_DIALECT_CIM_IR_CIMDIALECT
#define CIM22_DIALECT_CIM_IR_CIMDIALECT

include "mlir/IR/OpBase.td"

def CIM_Dialect : Dialect {
  let name = "cim";
  let cppNamespace = "::mlir::cim";
  let summary = "Logical operations for CIM computation";
  let description = [{
    The CIM dialect models logical compute operations independently of CIM22
    placement, physical data layout, NoC frames, and runtime behavior.
  }];
  let extraClassDeclaration = [{
    static StringRef getMatMulIntegerAttrName() {
      return "cim.onnx.matmul_integer";
    }
    static StringRef getConvIntegerAttrName() {
      return "cim.onnx.conv_integer";
    }
    static StringRef getTransactionIdxAttrName() { return "cim.transaction_idx"; }
    static StringRef getMatTileAttrName() { return "cim.m_tile"; }
    static StringRef getPlanBindingAttrName() { return "cim.plan_binding"; }
  }];
}

class CIM_Op<string mnemonic, list<Trait> traits = []>
    : Op<CIM_Dialect, mnemonic, traits>;

#endif // CIM22_DIALECT_CIM_IR_CIMDIALECT
