//===- CIMDevice.h - CIM22 hardware device boundary -----------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIM22_RUNTIME_CIMDEVICE_H
#define CIM22_RUNTIME_CIMDEVICE_H

#include "CIM22/Execution/CIMExecutable.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace cim22::runtime {

enum class CIMDevicePhase : uint8_t { Configuration, Work, Readback };

// One ordered CIM flit batch followed by an optional board pacing delay.
// Backends may split a batch into physical transfers, but must not reorder its
// flits or apply the delay before the complete batch is written.
struct CIMDeviceOperation {
  llvm::SmallVector<uint64_t> flits;
  uint32_t postDelayMs = 0;
  CIMDevicePhase phase = CIMDevicePhase::Configuration;
};

struct CIMDeviceReadbackCommand {
  size_t outputIndex = 0;
  execution::CIMDataType dataType = execution::CIMDataType::Int8;
  // Macro selection must settle before the request is sent.
  CIMDeviceOperation selection;
  CIMDeviceOperation request;
};

// One complete, transport-neutral CIM22 transaction. Device backends only
// package these operations for their physical link and recover response flits.
struct CIMDeviceProgram {
  std::vector<CIMDeviceOperation> operations;
  std::vector<CIMDeviceReadbackCommand> readbacks;
};

struct CIMDeviceReadback {
  size_t outputIndex = 0;
  uint64_t headerFlit = 0;
  std::array<uint64_t, 6> bodyFlits{};
};

struct CIMDeviceResponse {
  std::vector<CIMDeviceReadback> readbacks;
  // False includes manually managed clocks and unavailable optional serial.
  std::array<bool, 3> pllVerified{};
};

class CIMDevice {
public:
  virtual ~CIMDevice() = default;

  virtual llvm::Error open() = 0;
  virtual llvm::Error execute(const CIMDeviceProgram &program,
                              CIMDeviceResponse &response) = 0;
  // Inspect after close to include tail data/errors. Explicit open starts a
  // fresh receive session; it does not reset hardware or resume a failed run.
  virtual bool isFaulted() = 0;
  virtual void close() = 0;
};

} // namespace cim22::runtime

#endif // CIM22_RUNTIME_CIMDEVICE_H
