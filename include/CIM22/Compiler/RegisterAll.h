#ifndef CIM22_COMPILER_REGISTER_ALL_H
#define CIM22_COMPILER_REGISTER_ALL_H

#include "mlir/IR/BuiltinOps.h"
#include <utility>

namespace mlir {
class DialectRegistry;
}

/// Register CIM22 and required upstream dialects in the caller's registry.
void registerCIM22Dialects(mlir::DialectRegistry &registry);

/// Register external interface models used by CIM22 lowering.
void registerCIM22Extensions(mlir::DialectRegistry &registry);

/// Register CIM22 and upstream passes once per process.
void registerCIM22Passes();

/// Register the CIM22 pass pipelines and their named compiler stages.
void registerCIM22Pipelines();

/// Resolve the tagged Host and device modules in that order.
/// Report malformed splits on the root operation. The returned modules borrow
/// the root's storage, and callers verify their contents separately.
mlir::FailureOr<std::pair<mlir::ModuleOp, mlir::ModuleOp>>
getCIMSplitParts(mlir::ModuleOp root);

#endif
