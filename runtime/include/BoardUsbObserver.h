// Test-build-only capture of successful platform USB transfers.
#ifndef CIM22_RUNTIME_BOARDUSBOBSERVER_H
#define CIM22_RUNTIME_BOARDUSBOBSERVER_H

#include "UsbDriver.h"
#include <cstdint>

namespace cim22::runtime::detail {
using BoardUsbGate = int (*)(void *, uint64_t, uint64_t);

struct BoardUsbOptions {
  const char *directory;
  BoardUsbGate gate;
  void *context;
};

std::unique_ptr<CIMDevice>
createObservedUsbDevice(const UsbCIMDeviceConfig &config,
                        const BoardUsbOptions &options);
} // namespace cim22::runtime::detail

#endif
