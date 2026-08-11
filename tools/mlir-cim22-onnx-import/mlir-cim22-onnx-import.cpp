//===- mlir-cim22-onnx-import.cpp - Quantized ONNX importer -----*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIM22/Frontend/ONNX/Importer.h"

#include "llvm/Support/raw_ostream.h"

int main(int argc, char **argv) {
  if (argc != 2) {
    llvm::errs() << "usage: mlir-cim22-onnx-import <model.onnx>\n";
    return 1;
  }

  mlir::MLIRContext context;
  auto module = mlir::cim::importQuantizedONNX(context, argv[1]);
  if (failed(module))
    return 1;

  (*module)->print(llvm::outs());
  llvm::outs() << '\n';
  return 0;
}
