//===- PLLSerial.h - Optional board PLL control -------------*- C++ -*-===//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CIM22_RUNTIME_PLLSERIAL_H
#define CIM22_RUNTIME_PLLSERIAL_H

#include "CIM22/Runtime/CIMDevice.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>

namespace cim22::runtime {

struct PLLSerialConfig {
  std::string port;
  uint32_t configurationWord = 0;
  uint32_t workWord = 0;
  uint32_t readbackWord = 0;
};

// Checks raw commands against the table's 24 MHz reference and VCO floor;
// this is not a complete PLL electrical/stability specification.
llvm::Error validatePLLSerialConfig(const PLLSerialConfig &config);

// Sends three bytes MSB first over 9600/8N1 and requires their exact echo.
// An empty port selects the caller's manually configured clock. Linux also
// permits a missing port; Windows rejects nonempty ports. A false result means
// no serial verification, never a verified PLL configuration.
llvm::Expected<bool> configurePLL(const PLLSerialConfig &config,
                                  CIMDevicePhase phase);

} // namespace cim22::runtime

#endif // CIM22_RUNTIME_PLLSERIAL_H
