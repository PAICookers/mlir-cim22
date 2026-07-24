//===- mlir-cim22-opt.cpp - mlir-cim22 optimizer driver --------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIM22/Conversion/LinalgToCIM/Passes.h"
#include "CIM22/Dialect/CIM/IR/CIMDialect.h"
#include "CIM22/Dialect/CIMFrame/IR/CIMFrameDialect.h"
#include "CIM22/Dialect/CIMFrame/Transforms/Passes.h"
#include "CIM22/Target/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::cim::registerPasses();
  mlir::cimframe::registerPasses();
  mlir::cim22::target::registerPasses();

  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::cim::CIMDialect,
                  mlir::cimframe::CIMFrameDialect, mlir::func::FuncDialect,
                  mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect>();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "mlir-cim22 optimizer driver\n", registry));
}
