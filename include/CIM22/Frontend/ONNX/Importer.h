//===- Importer.h - Quantized ONNX frontend entry point ----------*- C++
//-*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIM22_FRONTEND_ONNX_IMPORTER_H
#define CIM22_FRONTEND_ONNX_IMPORTER_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

namespace onnx {
class ModelProto;
}

namespace mlir::cim {

FailureOr<OwningOpRef<ModuleOp>> importQuantizedONNX(MLIRContext &context,
                                                     StringRef filename);
FailureOr<OwningOpRef<ModuleOp>>
importQuantizedONNX(MLIRContext &context, const onnx::ModelProto &model);

} // namespace mlir::cim

#endif // CIM22_FRONTEND_ONNX_IMPORTER_H
