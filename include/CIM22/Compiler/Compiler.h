#ifndef CIM22_COMPILER_COMPILER_H
#define CIM22_COMPILER_COMPILER_H

#include "mlir/IR/BuiltinOps.h"
#include <utility>

namespace mlir {
class DialectRegistry;
}

/// Install CIM22 compiler dialects and interfaces in the caller's registry.
/// Optimizer and native compiler callers share one process-wide registration
/// of the passes and pipelines.
void registerCIM22Compiler(mlir::DialectRegistry &registry);

/// Register the CIM22 pass pipelines and their named compiler stages.
void registerCIM22Pipelines();

/// Resolve the tagged Host and device modules in that order.
/// Report malformed splits on the root operation. The returned modules borrow
/// the root's storage, and callers verify their contents separately.
mlir::FailureOr<std::pair<mlir::ModuleOp, mlir::ModuleOp>>
getCIMSplitParts(mlir::ModuleOp root);

#endif
