//===- CIMOps.td - CIM dialect operations ----------------*- tablegen -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIM22_DIALECT_CIM_IR_CIMOPS
#define CIM22_DIALECT_CIM_IR_CIMOPS

include "CIM22/Dialect/CIM/IR/CIMDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

def CIM_I21 : I<21>;

def CIM_VMMOp : CIM_Op<"vmm", [Pure]> {
  let summary = "Compute one logical INT8 VMM tile";
  let description = [{
    Computes 16 independent dot products between a 64-element input vector and
    the rows of a 16x64 weight tensor. Operands are interpreted as signed INT8
    values and each result is represented by a signless `i21` value.

    Before M5 scheduling this operation may carry zero-based `m_tile`,
    `n_tile`, and `k_tile` identity. M5 adds logical `work_id` and `group_id`.
    M4 then adds profile-local `core_idx` and `macro_idx` placement plus a
    verified `cim.mapping` semantic route result. These attributes do not model
    physical addresses, weight layout, frames, or runtime behavior.

    The first INT8 profile uses signed INT8 operands and the current signed-i21
    partial legality proof. Overflow behavior remains outside the deployable
    execution evidence owned by CTQ-020.
  }];

  let arguments = (ins RankedTensorOf<[I8]>:$input,
      RankedTensorOf<[I8]>:$weight);
  let results = (outs RankedTensorOf<[CIM_I21]>:$result);

  let assemblyFormat = [{
    $input `,` $weight attr-dict `:` type($input) `,` type($weight)
        `->` type($result)
  }];
  let hasVerifier = 1;
}

def CIM_StaticWeightOp
    : CIM_Op<"static_weight", [HasParent<"ModuleOp">, Symbol]> {
  let summary = "Define one immutable INT8 VMM weight tile";
  let arguments = (ins SymbolNameAttr:$sym_name,
      SignlessIntElementsAttr<8>:$value);
  let assemblyFormat = "$sym_name `=` $value attr-dict";
  let hasVerifier = 1;
}

def CIM_TransactionOp
    : CIM_Op<"transaction", [IsolatedFromAbove, HasParent<"func::FuncOp">,
      SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = "One synchronous interaction with the CIM chip";
  let description = [{
    A transaction owns the complete configuration, work, and output-cache
    readback sequence for one CIM execution wave. Host operations remain in
    the enclosing function and consume the explicit transaction results.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);
  let hasVerifier = 1;
}

def CIM_YieldOp : CIM_Op<"yield", [Terminator, HasParent<"TransactionOp">]> {
  let summary = "Return CIM readbacks from a transaction";
  let arguments = (ins Variadic<AnyType>:$values);
}

def CIM_ConfigureInputOp
    : CIM_Op<"configure_input"> {
  let summary = "Configure one mapped Macro input tile";
  let arguments = (ins RankedTensorOf<[I8]>:$input);
  let assemblyFormat = "$input attr-dict `:` type($input)";
  let hasVerifier = 1;
}

def CIM_ConfigureWeightOp
    : CIM_Op<"configure_weight"> {
  let summary = "Configure one mapped Macro static weight tile";
  let arguments = (ins SymbolRefAttr:$resource);
  let assemblyFormat = "$resource attr-dict";
  let hasVerifier = 1;
}

def CIM_DispatchOp : CIM_Op<"dispatch"> {
  let summary = "Describe one mapped VMM dispatch";
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def CIM_OnceOp : CIM_Op<"once"> {
  let summary = "Trigger one mapped core group once";
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def CIM_ReadbackOp : CIM_Op<"readback"> {
  let summary = "Read back one mapped i21 VMM partial";
  let results = (outs RankedTensorOf<[CIM_I21]>:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def CIM_GroupBarrierOp
    : CIM_Op<"group_barrier"> {
  let summary = "Complete one mapped execution-plan group";
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

#endif // CIM22_DIALECT_CIM_IR_CIMOPS
