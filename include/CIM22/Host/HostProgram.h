//===- HostProgram.h - Host/CIM program projection ------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIM22_HOST_HOSTPROGRAM_H
#define CIM22_HOST_HOSTPROGRAM_H

#include "CIM22/Dialect/CIM/IR/CIMOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"

#include <cstdint>
#include <variant>

namespace mlir::cim22::host {

enum class ViewDirection { HostToCIM, CIMToHost };

struct LogicalView {
  ViewDirection direction;
  RankedTensorType type;
  Value hostOwner;
  Operation *handoff;
  cim::GroupBarrierOp barrier;
  int64_t order;
  int64_t logicalSlot;
  int64_t workId;
};

struct HostStep {
  SmallVector<Operation *> operations;
};

struct CIMSegment {
  int64_t groupId;
  SmallVector<Operation *> operations;
  SmallVector<LogicalView> inputs;
  SmallVector<LogicalView> results;
  cim::GroupBarrierOp barrier;
};

using ProgramStep = std::variant<HostStep, CIMSegment>;

struct HostProgram {
  FlatSymbolRefAttr function;
  FunctionType type;
  SmallVector<ProgramStep> steps;
};

FailureOr<HostProgram> buildHostProgram(func::FuncOp function);

} // namespace mlir::cim22::host

#endif // CIM22_HOST_HOSTPROGRAM_H
