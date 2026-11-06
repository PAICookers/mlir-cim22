//===- Passes.h - Linalg to CIM passes -------------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIM22_CONVERSION_LINALGTOCIM_PASSES_H
#define CIM22_CONVERSION_LINALGTOCIM_PASSES_H

#include "CIM22/Dialect/CIM/IR/CIMDialect.h"
#include "CIM22/Dialect/CIM/IR/CIMOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"

namespace mlir::cim {
/// Track the source Linear across BF16 sign splitting and kernel outlining.
/// This temporary attribute scopes dispatch reordering to one source operator.
inline constexpr llvm::StringLiteral kResidencyLinearAttr =
    "cim.residency_linear";

#define GEN_PASS_DECL
#include "CIM22/Conversion/LinalgToCIM/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "CIM22/Conversion/LinalgToCIM/Passes.h.inc"
} // namespace mlir::cim

#endif // CIM22_CONVERSION_LINALGTOCIM_PASSES_H
