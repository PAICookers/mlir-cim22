//===- CIMFrameCodec.h - CIMFrame raw-flit encoding ------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIM22_TARGET_CIMFRAMECODEC_H
#define CIM22_TARGET_CIMFRAMECODEC_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

#include <cstdint>

namespace mlir::cim22::target {

// Encodes one verified, non-empty CIMFrame INT8 packet stage. The result is
// software-only until CTQ-020 supplies independent execution evidence.
FailureOr<SmallVector<uint64_t>> encodeCIMFrameInt8Packets(ModuleOp module);

} // namespace mlir::cim22::target

#endif // CIM22_TARGET_CIMFRAMECODEC_H
