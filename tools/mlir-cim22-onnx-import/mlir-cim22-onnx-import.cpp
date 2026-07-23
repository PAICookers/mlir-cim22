//===- mlir-cim22-onnx-import.cpp - Quantized ONNX importer -----*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIM22/Dialect/CIM/IR/CIMDialect.h"
#include "CIM22/Frontend/ONNX/Importer.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/Support/raw_ostream.h"

int main(int argc, char **argv) {
  if (argc != 2) {
    llvm::errs() << "usage: mlir-cim22-onnx-import <model.onnx>\n";
    return 1;
  }

  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::cim::CIMDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::linalg::LinalgDialect>();
  context.getOrLoadDialect<mlir::tensor::TensorDialect>();
  auto module = mlir::cim::importQuantizedONNX(context, argv[1]);
  if (failed(module))
    return 1;

  (*module)->print(llvm::outs());
  llvm::outs() << '\n';
  return 0;
}
