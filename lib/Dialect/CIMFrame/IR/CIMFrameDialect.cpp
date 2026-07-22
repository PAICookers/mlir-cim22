//===- CIMFrameDialect.cpp - CIMFrame dialect -----------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIM22/Dialect/CIMFrame/IR/CIMFrameDialect.h"
#include "CIM22/Dialect/CIMFrame/IR/CIMFrameOps.h"

using namespace mlir;
using namespace mlir::cimframe;

#include "CIM22/Dialect/CIMFrame/IR/CIMFrameOpsDialect.cpp.inc"

void CIMFrameDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "CIM22/Dialect/CIMFrame/IR/CIMFrameOps.cpp.inc"
      >();
}
