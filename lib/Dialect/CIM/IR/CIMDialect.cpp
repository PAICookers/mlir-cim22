//===- CIMDialect.cpp - CIM dialect ----------------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIM22/Dialect/CIM/IR/CIMDialect.h"
#include "CIM22/Dialect/CIM/IR/CIMOps.h"

using namespace mlir;
using namespace mlir::cim;

#include "CIM22/Dialect/CIM/IR/CIMOpsDialect.cpp.inc"

void CIMDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "CIM22/Dialect/CIM/IR/CIMOps.cpp.inc"
      >();
}
