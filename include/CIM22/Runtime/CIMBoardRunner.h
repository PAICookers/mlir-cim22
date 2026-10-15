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
#include "CIM22/Runtime/CIMDevice.h"

namespace cim22::runtime {

struct CIMBoardRunPolicy {
  uint32_t cacheWriteDelayMs = 1;
  uint32_t weightWriteDelayMs = 2;
  uint32_t workDelayMs = 2;
  // Board BF16 response packing is still a qualification experiment. Keep it
  // opt-in so a normal USB session cannot publish unverified BF16 values.
  bool allowCandidateBF16 = false;
};

class CIMBoardRunner final : public execution::CIMRunner {
public:
  CIMBoardRunner(CIMDevice &device, CIMBoardRunPolicy policy)
      : device_(device), policy_(policy) {}

  llvm::Error run(const execution::CIMTransaction &executable,
                  const execution::CIMRunInputs &inputs,
                  execution::CIMRunOutputs &outputs,
                  execution::CIMTraceSink *trace = nullptr) override;

private:
  CIMDevice &device_;
  CIMBoardRunPolicy policy_;
};

} // namespace cim22::runtime

#endif // CIM22_RUNTIME_CIMBOARDRUNNER_H
