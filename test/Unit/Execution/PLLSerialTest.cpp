//===- PLLSerialTest.cpp - PLL configuration and serial tests ------------===//
// No physical serial device is opened by these tests.

#include "CIM22/Runtime/PLLSerial.h"

#include "llvm/Support/Error.h"

#include <array>
#include <cassert>
#include <string>
#ifndef _WIN32
#include <chrono>
#include <cstdlib>
#include <fcntl.h>
#include <poll.h>
#include <pty.h>
#include <sys/stat.h>
#include <termios.h>
#include <thread>
#include <unistd.h>
#endif

using namespace cim22::runtime;

namespace {
#ifndef _WIN32
struct Pty {
  int master = -1;
  int slave = -1;
  char name[128]{};
  Pty() { assert(::openpty(&master, &slave, name, nullptr, nullptr) == 0); }
  ~Pty() {
    ::close(master);
    ::close(slave);
  }
};

void exchange(CIMDevicePhase phase, uint32_t expected, bool echo) {
  Pty pty;
  PLLSerialConfig config{pty.name, 0x2828a4, 0x2c28a4, 0x3028a4};
  std::thread peer([&] {
    std::array<uint8_t, 3> request{};
    size_t offset = 0;
    while (offset < request.size()) {
      pollfd descriptor{pty.master, POLLIN, 0};
      assert(::poll(&descriptor, 1, 1000) == 1);
      const ssize_t count =
          ::read(pty.master, request.data() + offset, request.size() - offset);
      assert(count > 0);
      offset += count;
    }
    const std::array<uint8_t, 3> bytes{static_cast<uint8_t>(expected >> 16),
                                       static_cast<uint8_t>(expected >> 8),
                                       static_cast<uint8_t>(expected)};
    assert(request == bytes);
    termios actual{};
    assert(::tcgetattr(pty.slave, &actual) == 0);
    assert(::cfgetispeed(&actual) == B9600 && ::cfgetospeed(&actual) == B9600);
    assert((actual.c_cflag & (PARENB | CSTOPB | CRTSCTS)) == 0);
    assert((actual.c_cflag & CSIZE) == CS8);
    if (!echo) {
      request.back() ^= 1;
      assert(::write(pty.master, request.data(), request.size()) == 3);
      return;
    }
    // Noise and overlapping prefixes must not become a partial-echo ACK.
    const std::array<uint8_t, 2> noise{0xff, request.front()};
    assert(::write(pty.master, noise.data(), noise.size()) == 2);
    for (uint8_t byte : request) {
      assert(::write(pty.master, &byte, 1) == 1);
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  });
  auto result = configurePLL(config, phase);
  peer.join();
  if (echo) {
    assert(result && *result);
  } else {
    assert(!result);
    assert(llvm::toString(result.takeError()).find("echo timed out") !=
           std::string::npos);
  }
}

void testMissingPortFallback() {
  char directory[] = "/tmp/cim22-pll-test-XXXXXX";
  assert(::mkdtemp(directory));
  const std::string missing = std::string(directory) + "/missing";
  PLLSerialConfig config{missing, 0x2828a4, 0x2828a4, 0x2828a4};
  auto unavailable = configurePLL(config, CIMDevicePhase::Work);
  assert(unavailable && !*unavailable);
  assert(::rmdir(directory) == 0);
}

void testPresentPortErrorsDoNotFallback() {
  PLLSerialConfig config{"/dev/null", 0x2828a4, 0x2828a4, 0x2828a4};
  auto notSerial = configurePLL(config, CIMDevicePhase::Configuration);
  assert(!notSerial);
  assert(llvm::toString(notSerial.takeError()).find("serial parameters") !=
         std::string::npos);
  if (::geteuid() != 0) {
    Pty pty;
    struct stat status{};
    assert(::fstat(pty.slave, &status) == 0);
    assert(::chmod(pty.name, 0) == 0);
    config.port = pty.name;
    auto denied = configurePLL(config, CIMDevicePhase::Configuration);
    assert(::chmod(pty.name, status.st_mode & 0777) == 0);
    assert(!denied);
    assert(llvm::toString(denied.takeError()).find("permissions") !=
           std::string::npos);
  }
}
#endif

void testManualClock() {
  for (auto phase : {CIMDevicePhase::Configuration, CIMDevicePhase::Work,
                     CIMDevicePhase::Readback}) {
    auto omitted = configurePLL({}, phase);
    assert(omitted && !*omitted);
  }
}

#ifdef _WIN32
void testSerialControlIsUnsupported() {
  PLLSerialConfig config{"COM999", 0x2828a4, 0x2828a4, 0x2828a4};
  auto result = configurePLL(config, CIMDevicePhase::Configuration);
  assert(!result);
  assert(llvm::toString(result.takeError()).find("supported only on Linux") !=
         std::string::npos);
}
#endif

void testInvalidCommands() {
  for (uint32_t word : {0u, 0x1000000u}) {
    PLLSerialConfig config{"/missing", word, 0x2828a4, 0x2828a4};
    auto result = configurePLL(config, CIMDevicePhase::Configuration);
    assert(!result);
    llvm::consumeError(result.takeError());
  }
}
} // namespace

int main() {
  testManualClock();
  testInvalidCommands();
#ifdef _WIN32
  testSerialControlIsUnsupported();
#else
  exchange(CIMDevicePhase::Configuration, 0x2828a4, true);
  exchange(CIMDevicePhase::Work, 0x2c28a4, true);
  exchange(CIMDevicePhase::Readback, 0x3028a4, true);
  exchange(CIMDevicePhase::Configuration, 0x2828a4, false);
  testMissingPortFallback();
  testPresentPortErrorsDoNotFallback();
#endif
}
