//===- CIMSegments.h - CIM execution segments -----------------*- C++ -*-===//
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

/// Non-owning view of one verified Host-visible CIM launch boundary.
struct CIMSegmentInfo {
  int64_t segmentId;
  SmallVector<Operation *> operations;
  SmallVector<Value> inputs;
  SmallVector<Value> outputs;
};

/// Collects segments from a function that passed operation and plan
/// verification, preserving execution-plan order without taking ownership.
SmallVector<CIMSegmentInfo> analyzeCIMSegments(func::FuncOp function);

} // namespace mlir::cim

#endif // CIM22_CONVERSION_LINALGTOCIM_CIMSEGMENTS_H
