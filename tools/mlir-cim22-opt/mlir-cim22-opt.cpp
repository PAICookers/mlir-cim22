//===- mlir-cim22-opt.cpp - mlir-cim22 optimizer driver --------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIM22/Compiler/Compiler.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registerCIM22Compiler(registry);
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "mlir-cim22 optimizer driver\n", registry));
}
