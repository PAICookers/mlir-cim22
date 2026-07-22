//===- Passes.h - CIMFrame passes ------------------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIM22_DIALECT_CIMFRAME_TRANSFORMS_PASSES_H
#define CIM22_DIALECT_CIMFRAME_TRANSFORMS_PASSES_H

#include "CIM22/Dialect/CIMFrame/IR/CIMFrameOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::cimframe {
#define GEN_PASS_DECL
#include "CIM22/Dialect/CIMFrame/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "CIM22/Dialect/CIMFrame/Transforms/Passes.h.inc"
} // namespace mlir::cimframe

#endif // CIM22_DIALECT_CIMFRAME_TRANSFORMS_PASSES_H
