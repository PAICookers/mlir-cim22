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
#ifdef _WIN32
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Windows/WindowsSupport.h"
#endif

int main(int argc, char **argv) {
#ifdef _WIN32
  // MlirOptMain parses CLI paths before its InitLLVM converts Windows argv.
  llvm::BumpPtrAllocator allocator;
  llvm::SmallVector<const char *, 32> arguments;
  if (auto error =
          llvm::sys::windows::GetCommandLineArguments(arguments, allocator)) {
    llvm::errs() << error.message() << "\n";
    return 1;
  }
  arguments.push_back(nullptr);
  argc = arguments.size() - 1;
  argv = const_cast<char **>(arguments.data());
#endif
  mlir::DialectRegistry registry;
  registerCIM22Compiler(registry);
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "mlir-cim22 optimizer driver\n", registry));
}
