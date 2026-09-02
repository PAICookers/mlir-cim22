//===- CIMUartRunner.cpp - CIM22 UART runner boundary ----------*- C++ -*-===//

#include "CIM22/Execution/CIMRunner.h"

#include "llvm/Support/Error.h"

namespace cim22::execution {

llvm::Error CIMUartRunner::run(const CIMTransaction &, const CIMRunInputs &,
                               CIMRunOutputs &, CIMTraceSink *) {
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "CIM UART transport is not implemented for this target profile");
}

} // namespace cim22::execution
