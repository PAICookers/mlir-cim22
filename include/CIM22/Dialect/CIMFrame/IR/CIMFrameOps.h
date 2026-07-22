//===- CIMFrameOps.h - CIMFrame operations --------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIM22_DIALECT_CIMFRAME_IR_CIMFRAMEOPS_H
#define CIM22_DIALECT_CIMFRAME_IR_CIMFRAMEOPS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "CIM22/Dialect/CIMFrame/IR/CIMFrameOps.h.inc"

#endif // CIM22_DIALECT_CIMFRAME_IR_CIMFRAMEOPS_H
