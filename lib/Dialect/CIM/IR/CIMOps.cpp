//===- CIMOps.cpp - CIM dialect operations ---------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIM22/Dialect/CIM/IR/CIMOps.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::cim;

LogicalResult VMMOp::verify() {
  RankedTensorType inputType = getInput().getType();
  RankedTensorType weightType = getWeight().getType();
  RankedTensorType resultType = getResult().getType();

  if (inputType.getShape() != llvm::ArrayRef<int64_t>{64})
    return emitOpError("expects input shape [64], but got ") << inputType;
  if (weightType.getShape() != llvm::ArrayRef<int64_t>{16, 64})
    return emitOpError(
               "expects weight shape [16, 64] in [N, K] order, but got ")
           << weightType;
  if (resultType.getShape() != llvm::ArrayRef<int64_t>{16})
    return emitOpError("expects result shape [16], but got ") << resultType;
  return success();
}

#define GET_OP_CLASSES
#include "CIM22/Dialect/CIM/IR/CIMOps.cpp.inc"
