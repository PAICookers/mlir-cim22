// Synchronous C ABI adapter shared by USB dispatch and its native test fixture.
#ifndef CIM22_RUNTIME_CLOCKCALLBACK_H
#define CIM22_RUNTIME_CLOCKCALLBACK_H
#include "CIM22/Runtime/CIMDevice.h"
#include "CIM22/Runtime/Runtime.h"
#include "llvm/Support/Error.h"
#include <array>
#include <functional>
namespace cim22::runtime::detail {
inline std::function<llvm::Expected<bool>(CIMDevicePhase)>
makeClockCallback(const CIM22ClockControl *clock) {
  if (!clock)
    return {};
  return [control = *clock](CIMDevicePhase phase) -> llvm::Expected<bool> {
    std::array<char, 1024> error{};
    uint8_t verified = 0;
    if (!control.callback ||
        control.callback(control.context, static_cast<uint32_t>(phase),
                         &verified, error.data(), error.size())) {
      error.back() = 0;
      return llvm::createStringError(llvm::inconvertibleErrorCode(), "%s",
                                     error[0] ? error.data()
                                              : "clock callback failed");
    }
    return verified != 0;
  };
}
} // namespace cim22::runtime::detail
#endif
