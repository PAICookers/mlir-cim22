//===- CIMOps.h - CIM dialect operations ----------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIM22_DIALECT_CIM_IR_CIMOPS_H
#define CIM22_DIALECT_CIM_IR_CIMOPS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "CIM22/Dialect/CIM/IR/CIMOps.h.inc"

#endif // CIM22_DIALECT_CIM_IR_CIMOPS_H
