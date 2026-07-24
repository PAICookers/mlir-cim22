//===- Passes.h - CIM22 target passes --------------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIM22_TARGET_PASSES_H
#define CIM22_TARGET_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir::cim22::target {
#define GEN_PASS_DECL
#include "CIM22/Target/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "CIM22/Target/Passes.h.inc"
} // namespace mlir::cim22::target

#endif // CIM22_TARGET_PASSES_H
