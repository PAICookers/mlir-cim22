#ifndef CIM22_COMPILER_COMPILER_H
#define CIM22_COMPILER_COMPILER_H

#include "mlir/IR/BuiltinOps.h"
#include <utility>

namespace mlir {
class DialectRegistry;
}

// Shared optimizer/JIT registration. Pass registration is process-wide and
// performed once; dialect interfaces belong to each caller's registry.
void registerCIM22Compiler(mlir::DialectRegistry &registry);
void registerCIM22Pipelines();

// Both pass pipelines and native compilation require the same split contract.
mlir::FailureOr<std::pair<mlir::ModuleOp, mlir::ModuleOp>>
getCIMSplitParts(mlir::ModuleOp root);

#endif
