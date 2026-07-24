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

def CIM_I21 : I<21>;

def CIM_VMMOp : CIM_Op<"vmm", [Pure]> {
  let summary = "Compute one logical INT8 VMM tile";
  let description = [{
    Computes 16 independent dot products between a 64-element input vector and
    the rows of a 16x64 weight tensor. Operands are interpreted as signed INT8
    values and each result is represented by a signless `i21` value.

    Before M5 scheduling this operation may carry zero-based `m_tile`,
    `n_tile`, and `k_tile` identity. M5 adds logical `work_id` and `group_id`.
    M4 then adds profile-local `core_slot` and `macro_slot` placement plus a
    verified `cim.mapping` semantic route result. These attributes do not model
    physical addresses, weight layout, frames, or runtime behavior.

    TODO(CTQ-013): Confirm the signed INT8 input domain.
    FIXME(CTQ-013): Overflow behavior is not a deployable contract.
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

#endif // CIM22_DIALECT_CIM_IR_CIMOPS
