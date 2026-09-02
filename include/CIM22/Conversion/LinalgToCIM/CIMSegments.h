//===- CIMSegments.h - CIM transaction analysis ----------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIM22_CONVERSION_LINALGTOCIM_CIMSEGMENTS_H
#define CIM22_CONVERSION_LINALGTOCIM_CIMSEGMENTS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace mlir::cim {

/// Non-owning view of one verified Host-visible CIM transaction.
struct CIMTransactionInfo {
  int64_t transactionIdx;
  SmallVector<Value> inputs;
  SmallVector<Value> outputs;
};

/// Identifies the CIM operations that form an execution plan.
bool isExecutionPlanOp(Operation *op);

/// Collects transactions from a verified function in execution order.
SmallVector<CIMTransactionInfo> analyzeCIMTransactions(func::FuncOp function);

} // namespace mlir::cim

#endif // CIM22_CONVERSION_LINALGTOCIM_CIMSEGMENTS_H
