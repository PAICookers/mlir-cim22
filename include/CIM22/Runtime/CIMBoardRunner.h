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

// Supplier-compatible default pacing in CIM22HardwareConfig field order.
inline CIM22HardwareConfig defaultCIM22HardwareConfig() {
  return {20, 1, 2, 10, 40};
}

// Materializes validated device commands and returns decoded readbacks in
// binding order. The open device is borrowed and must outlive the runner.
class CIMBoardRunner final : public execution::CIMRunner {
public:
  CIMBoardRunner(CIMDevice &device, CIM22HardwareConfig hardware)
      : device_(device), hardware_(hardware) {}

  llvm::Error run(const execution::CIMTransaction &executable,
                  const execution::CIMRunInputs &inputs,
                  execution::CIMRunOutputs &outputs,
                  execution::CIMTraceSink *trace = nullptr) override;

  // Applies the requested BF16 compensation on the Host.
  llvm::Error run(const execution::CIMTransaction &executable,
                  const execution::CIMRunInputs &inputs,
                  execution::CIMRunOutputs &outputs,
                  execution::BF16Compensation compensation,
                  execution::CIMTraceSink *trace = nullptr);

  // Capture validated headers and six raw body flits per readback, with
  // outputIndex identifying the binding.
  llvm::Expected<CIMDeviceResponse>
  capture(const execution::CIMTransaction &executable,
          const execution::CIMRunInputs &inputs,
          execution::CIMTraceSink *trace = nullptr);

private:
  llvm::Expected<CIMDeviceResponse>
  captureImpl(const execution::CIMTransaction &executable,
              const execution::CIMRunInputs &inputs,
              execution::CIMTraceSink *trace, uint64_t &sequence,
              bool zeroWeightExponent, bool captureRawReadbacks);

  CIMDevice &device_;
  CIM22HardwareConfig hardware_;
};

} // namespace cim22::runtime

#endif // CIM22_RUNTIME_CIMBOARDRUNNER_H
