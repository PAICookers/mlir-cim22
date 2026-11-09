//===- PLLSerial.cpp - Optional Linux board PLL control -----------------===//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "CIM22/Runtime/PLLSerial.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstring>
#ifndef _WIN32
#include <fcntl.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>
#endif
#include <thread>
#include <utility>

namespace cim22::runtime {
namespace {
using Clock = std::chrono::steady_clock;

#ifndef _WIN32
llvm::Error serialError(const char *operation) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), "PLL %s: %s",
                                 operation, std::strerror(errno));
}

llvm::Expected<bool> waitPort(int port, short events,
                              Clock::time_point deadline) {
  while (Clock::now() < deadline) {
    const auto remaining =
        std::chrono::duration_cast<std::chrono::milliseconds>(deadline -
                                                              Clock::now());
    pollfd descriptor{port, events, 0};
    const int result = ::poll(&descriptor, 1,
                              std::max(1, static_cast<int>(remaining.count())));
    if (result < 0) {
      if (errno == EINTR)
        continue;
      return serialError("serial wait failed");
    }
    if (descriptor.revents & (POLLERR | POLLHUP | POLLNVAL))
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "PLL serial port disconnected");
    if (descriptor.revents & events)
      return true;
  }
  return false;
}
#endif
} // namespace

llvm::Error validatePLLSerialConfig(const PLLSerialConfig &config) {
  if (config.port.empty())
    return llvm::Error::success();
  if (config.port.find('\0') != std::string::npos)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "PLL port contains an embedded NUL");
  for (uint32_t word :
       {config.configurationWord, config.workWord, config.readbackWord}) {
    if (word > 0xffffff)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "PLL command exceeds three bytes");
    // Raw C/C++ callers can bypass Python's preset table and validation.
    const uint32_t nf = ((word >> 18) & 0x3f) + 1;
    const uint32_t nr = ((word >> 14) & 0xf) + 1;
    if (24 * nf < 250 * nr)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "PLL VCO must be at least 250 MHz");
  }
  return llvm::Error::success();
}

llvm::Expected<bool> configurePLL(const PLLSerialConfig &config,
                                  CIMDevicePhase phase) {
  if (llvm::Error error = validatePLLSerialConfig(config))
    return std::move(error);
  if (config.port.empty())
    return false;
#ifdef _WIN32
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "PLL serial control is currently supported only on Linux; omit the "
      "serial port and configure the clock manually on Windows");
#else
  uint32_t word;
  switch (phase) {
  case CIMDevicePhase::Configuration:
    word = config.configurationWord;
    break;
  case CIMDevicePhase::Work:
    word = config.workWord;
    break;
  case CIMDevicePhase::Readback:
    word = config.readbackWord;
    break;
  default:
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "invalid PLL phase");
  }
  const int port =
      ::open(config.port.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK | O_CLOEXEC);
  if (port < 0) {
    if (errno == ENOENT || errno == ENODEV)
      return false;
    return serialError("cannot open serial port (check permissions)");
  }
  auto closePort = llvm::make_scope_exit([&] { ::close(port); });
  termios parameters{};
  if (::tcgetattr(port, &parameters) != 0)
    return serialError("cannot read serial parameters");
  ::cfmakeraw(&parameters);
  parameters.c_cflag &= ~(CSIZE | PARENB | CSTOPB | CRTSCTS | HUPCL);
  parameters.c_cflag |= CS8 | CLOCAL | CREAD;
  parameters.c_iflag &= ~(IXON | IXOFF | IXANY);
  parameters.c_cc[VMIN] = 0;
  parameters.c_cc[VTIME] = 0;
  if (::cfsetispeed(&parameters, B9600) != 0 ||
      ::cfsetospeed(&parameters, B9600) != 0 ||
      ::tcsetattr(port, TCSANOW, &parameters) != 0)
    return serialError("cannot configure 9600/8N1");
  int modemLines = TIOCM_RTS | TIOCM_DTR;
  if (::ioctl(port, TIOCMBIC, &modemLines) != 0 && errno != ENOTTY)
    return serialError("cannot disable RTS/DTR");
  if (::tcflush(port, TCIOFLUSH) != 0)
    return serialError("cannot clear serial queues");

  const std::array<uint8_t, 3> bytes{static_cast<uint8_t>(word >> 16),
                                     static_cast<uint8_t>(word >> 8),
                                     static_cast<uint8_t>(word)};
  const auto deadline = Clock::now() + std::chrono::milliseconds(200);
  size_t offset = 0;
  while (offset < bytes.size()) {
    auto ready = waitPort(port, POLLOUT, deadline);
    if (!ready)
      return ready.takeError();
    if (!*ready)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "PLL serial write timed out");
    const ssize_t count =
        ::write(port, bytes.data() + offset, bytes.size() - offset);
    if (count < 0) {
      if (errno == EINTR || errno == EAGAIN)
        continue;
      return serialError("serial write failed");
    }
    offset += static_cast<size_t>(count);
    if (count == 0)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  std::array<uint8_t, 3> window{};
  size_t received = 0;
  while (true) {
    auto ready = waitPort(port, POLLIN, deadline);
    if (!ready)
      return ready.takeError();
    if (!*ready)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "PLL echo timed out without a matching reply");
    uint8_t reply[256];
    const ssize_t count = ::read(port, reply, sizeof(reply));
    if (count < 0) {
      if (errno == EINTR || errno == EAGAIN)
        continue;
      return serialError("serial read failed");
    }
    for (ssize_t index = 0; index < count; ++index) {
      window[0] = window[1];
      window[1] = window[2];
      window[2] = reply[index];
      if (++received >= bytes.size() && window == bytes)
        return true;
    }
  }
#endif
}
} // namespace cim22::runtime
