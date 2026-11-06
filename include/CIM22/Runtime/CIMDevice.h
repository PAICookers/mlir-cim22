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

// One ordered CIM flit batch followed by an optional pacing delay in ms.
// Backends may split a batch into physical transfers, but must not reorder its
// flits or apply the delay before the complete batch is written.
struct CIMDeviceOperation {
  llvm::SmallVector<uint64_t> flits;
  uint32_t postDelayMs = 0;
  CIMDevicePhase phase = CIMDevicePhase::Configuration;
};

// The response head supplies Core and Cache row; outputIndex refers to the
// caller's binding order. Macro identity comes from the enclosing batch.
struct CIMDeviceReadbackTarget {
  size_t outputIndex = 0;
  uint8_t coreIndex = 0;
  uint8_t cacheAddress = 0;
};

// Select one Macro on the participating Cores, wait for selection to settle,
// send the requests, and collect exactly one reply per target in any order.
// Drain this set before another selection changes the meaning of its keys.
struct CIMDeviceReadbackBatch {
  CIMDeviceOperation selection;
  CIMDeviceOperation request;
  std::vector<CIMDeviceReadbackTarget> targets;
};

// One transport-neutral CIM22 transaction. Configuration and Work operations
// run before readback batches, in their listed order.
struct CIMDeviceProgram {
  std::vector<CIMDeviceOperation> operations;
  std::vector<CIMDeviceReadbackBatch> readbackBatches;
  // Diagnostic capture retains raw body padding. Headers and reply coverage
  // still have to match the readback plan.
  bool captureRawReadbacks = false;
};

struct CIMDeviceReadback {
  size_t outputIndex = 0;
  uint64_t headerFlit = 0;
  std::array<uint64_t, 6> bodyFlits{};
};

struct CIMDeviceResponse {
  // May be in packet-completion order. Use outputIndex to locate bindings.
  std::vector<CIMDeviceReadback> readbacks;
  // False includes manually managed clocks and unavailable optional serial.
  // True may reuse an earlier verified PLL word in the same open session.
  // Entries follow CIMDevicePhase order.
  std::array<bool, 3> pllVerified{};
};

class CIMDevice {
public:
  virtual ~CIMDevice() = default;

  // Begin a fresh receive session. Hardware recovery is a separate operation.
  virtual llvm::Error open() = 0;
  // Synchronous. On success, response contains the complete readback set.
  // Treat response contents as unspecified after a failure. A transport
  // fault blocks subsequent execution until an explicit open.
  virtual llvm::Error execute(const CIMDeviceProgram &program,
                              CIMDeviceResponse &response) = 0;
  // Inspect after close to include tail data and receive errors.
  virtual bool isFaulted() = 0;
  // Retire receive operations and release the link, preserving fault evidence.
  virtual void close() = 0;
};

} // namespace cim22::runtime

#endif // CIM22_RUNTIME_CIMDEVICE_H
