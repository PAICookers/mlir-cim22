//===- CIMBoardRunner.h - CIM22 hardware runner ---------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIM22_RUNTIME_CIMBOARDRUNNER_H
#define CIM22_RUNTIME_CIMBOARDRUNNER_H

#include "CIM22/Execution/CIMRunner.h"
#include "CIM22/Execution/Runtime.h"
#include "CIM22/Runtime/CIMDevice.h"

namespace cim22::runtime {

inline CIM22HardwareConfig defaultCIM22HardwareConfig() {
  return {20, 1, 2, 10, 40};
}

class CIMBoardRunner final : public execution::CIMRunner {
public:
  CIMBoardRunner(CIMDevice &device, CIM22HardwareConfig hardware)
      : device_(device), hardware_(hardware) {}

  llvm::Error run(const execution::CIMTransaction &executable,
                  const execution::CIMRunInputs &inputs,
                  execution::CIMRunOutputs &outputs,
                  execution::CIMTraceSink *trace = nullptr) override;

  llvm::Error run(const execution::CIMTransaction &executable,
                  const execution::CIMRunInputs &inputs,
                  execution::CIMRunOutputs &outputs,
                  execution::BF16Compensation compensation,
                  execution::CIMTraceSink *trace = nullptr);

  // Capture validated response headers and six raw body flits per readback.
  llvm::Expected<CIMDeviceResponse>
  capture(const execution::CIMTransaction &executable,
          const execution::CIMRunInputs &inputs,
          execution::CIMTraceSink *trace = nullptr);

private:
  llvm::Expected<CIMDeviceResponse>
  captureImpl(const execution::CIMTransaction &executable,
              const execution::CIMRunInputs &inputs,
              execution::CIMTraceSink *trace, uint64_t &sequence,
              bool zeroWeightExponent = false);

  CIMDevice &device_;
  CIM22HardwareConfig hardware_;
};

} // namespace cim22::runtime

#endif // CIM22_RUNTIME_CIMBOARDRUNNER_H
