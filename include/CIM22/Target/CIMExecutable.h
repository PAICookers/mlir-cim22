//===- CIMExecutable.h - CIM22 target executable compiler ------*- C++ -*-===//

#ifndef CIM22_TARGET_CIMEXECUTABLE_H
#define CIM22_TARGET_CIMEXECUTABLE_H

#include "CIM22/Execution/CIMExecutable.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::cim22::target {

mlir::FailureOr<::cim22::execution::CIMExecutable>
compileCIMExecutable(mlir::ModuleOp verifiedModule);

} // namespace mlir::cim22::target

#endif // CIM22_TARGET_CIMEXECUTABLE_H
