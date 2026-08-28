//===- CIMSegments.cpp - CIM execution segments ---------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIM22/Conversion/LinalgToCIM/CIMSegments.h"

#include "CIM22/Dialect/CIM/IR/CIMDialect.h"
#include "CIM22/Dialect/CIM/IR/CIMOps.h"

#include "llvm/ADT/SetVector.h"

namespace mlir::cim {
namespace {
bool isExecutionPlanOp(Operation *op) {
  return isa<ConfigureInputOp, ConfigureWeightOp, DispatchOp, OnceOp,
             ReadbackOp, GroupBarrierOp>(op);
}
} // namespace

SmallVector<CIMSegmentInfo> analyzeCIMSegments(func::FuncOp function) {
  SmallVector<CIMSegmentInfo> segments;
  llvm::SetVector<Value> inputs;
  llvm::SetVector<Value> outputs;

  auto finishSegment = [&] {
    if (segments.empty())
      return;
    segments.back().inputs.assign(inputs.begin(), inputs.end());
    segments.back().outputs.assign(outputs.begin(), outputs.end());
    inputs.clear();
    outputs.clear();
  };

  for (Operation &op : function.getBody().front()) {
    if (!isExecutionPlanOp(&op))
      continue;
    int64_t segmentId =
        cast<IntegerAttr>(op.getAttr(CIMDialect::getSegmentIdAttrName()))
            .getInt();
    if (segments.empty() || segments.back().segmentId != segmentId) {
      finishSegment();
      segments.push_back(CIMSegmentInfo{segmentId, {}, {}, {}});
    }
    segments.back().operations.push_back(&op);
    if (auto input = dyn_cast<ConfigureInputOp>(op))
      inputs.insert(input.getInput());
    if (auto readback = dyn_cast<ReadbackOp>(op))
      outputs.insert(readback.getResult());
  }
  finishSegment();
  return segments;
}

} // namespace mlir::cim
