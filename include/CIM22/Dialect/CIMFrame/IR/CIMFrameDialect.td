//===- CIMFrameDialect.td - CIMFrame dialect -------------*- tablegen -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIM22_DIALECT_CIMFRAME_IR_CIMFRAMEDIALECT
#define CIM22_DIALECT_CIMFRAME_IR_CIMFRAMEDIALECT

include "mlir/IR/OpBase.td"

def CIMFrame_Dialect : Dialect {
  let name = "cimframe";
  let cppNamespace = "::mlir::cimframe";
  let summary = "Typed CIM packet-stage commands and packets";
  let description = [{
    The CIMFrame dialect separates typed packet-stage commands from typed
    NoC packet semantics. M2.1 models only an INT8 start-once control prefix;
    it has no flit, byte, artifact, or runtime translation.
  }];
}

class CIMFrame_Op<string mnemonic, list<Trait> traits = []>
    : Op<CIMFrame_Dialect, mnemonic, traits>;

#endif // CIM22_DIALECT_CIMFRAME_IR_CIMFRAMEDIALECT
