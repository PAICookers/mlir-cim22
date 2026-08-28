//===- ExecutionPlanVerifier.h - CIM execution-plan checks -*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIM22_CONVERSION_LINALGTOCIM_EXECUTIONPLANVERIFIER_H
#define CIM22_CONVERSION_LINALGTOCIM_EXECUTIONPLANVERIFIER_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/LogicalResult.h"

namespace mlir::cim {

/// Verifies cross-operation invariants of an already MLIR-verified plan.
///
/// Callers must run the ordinary operation verifier first. This routine owns
/// only plan-wide ordering, identity, and Host-boundary checks.
LogicalResult verifyCIMExecutionPlan(func::FuncOp function);

} // namespace mlir::cim

#endif // CIM22_CONVERSION_LINALGTOCIM_EXECUTIONPLANVERIFIER_H
