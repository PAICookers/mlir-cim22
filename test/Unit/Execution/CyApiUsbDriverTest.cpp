// Tests the real adapter against a controlled SDK, without a USB board.
#include "CIM22/Runtime/Detail/UsbDriver.h"
#include <windows.h>

// Match the official SDK's prerequisite include order.
#include <CyAPI.h>
#include <cassert>
#include <iostream>

using namespace cim22::runtime;
using namespace cim22::runtime::detail;
namespace fake = fake_cyapi;

static UsbCIMDeviceConfig config() {
  UsbCIMDeviceConfig c;
  c.vendorId = 0x04b4;
  c.productId = 0x00f1;
  c.bulkInEndpoint = 0x81;
  c.bulkOutEndpoint = 0x01;
  c.receivePollTimeoutMs = 1;
  c.receiveChunkBytes = 64;
  return c;
}
static void expectError(llvm::Error error, const char *part) {
  assert(error);
  assert(llvm::toString(std::move(error)).find(part) != std::string::npos);
}

int main() {
  // Selection must follow the profile, not blindly open device zero.
  fake::reset();
  fake::devices.insert(fake::devices.begin(), {0x1234, 0x5678, 2, 0});
  auto driver = createPlatformUsbDriver();
  llvm::cantFail(driver->open(config()));
  assert(fake::selected == 1 && fake::begins == 1);
  uint8_t bytes[64]{};
  assert(llvm::cantFail(driver->write(bytes)) == 3);
  uint8_t largeWrite[1024]{};
  assert(llvm::cantFail(driver->write(largeWrite)) == 3);
  assert(fake::requestedWrite == 512);
  // Polls retain the exact outstanding request and its storage.
  auto *pending = fake::pending;
  assert(llvm::cantFail(driver->read(bytes)) == 0);
  assert(llvm::cantFail(driver->read(bytes)) == 0);
  assert(fake::pending == pending && fake::begins == 1 && !fake::aborts);
  fake::complete(7);
  assert(llvm::cantFail(driver->read(bytes)) == 7 && bytes[6] == 0x5a);
  assert(fake::finishes == 1);
  assert(llvm::cantFail(driver->read(bytes)) == 0);
  driver->cancel();
  driver->close();
  assert(fake::aborts == 1 && fake::finishes == 2 && !fake::pending);
  driver->close(); // Idempotent; no double Finish or event destruction.

  for (int scenario = 0; scenario != 9; ++scenario) {
    fake::reset();
    auto c = config();
    const char *error = nullptr;
    switch (scenario) {
    case 0:
      fake::devices.clear();
      error = "not found";
      break;
    case 1:
      fake::devices.push_back({});
      error = "ambiguous";
      break;
    case 2:
      c.busNumber = 1;
      error = "bus";
      break;
    case 3:
      c.interfaceNumber = 1;
      error = "interface";
      break;
    case 4:
      fake::missingEndpoint = true;
      error = "endpoint";
      break;
    case 5:
      fake::beginFails = true;
      error = "BeginDataXfer";
      break;
    case 6:
      fake::altFails = true;
      error = "interface";
      break;
    case 7:
      c.receiveChunkBytes = 0;
      error = "receiveChunkBytes";
      break;
    case 8:
      c.receiveChunkBytes = 4 * 1024 * 1024 + 1;
      error = "receiveChunkBytes";
      break;
    }
    expectError(driver->open(c), error);
    driver->close();
    assert(!fake::pending);
  }

  fake::reset();
  fake::devices.push_back({0x04b4, 0x00f1, 2, 0});
  auto c = config();
  c.deviceAddress = 2;
  llvm::cantFail(driver->open(c));
  assert(fake::selected == 1);
  expectError(driver->write({}).takeError(), "empty");
  expectError(driver->read(llvm::MutableArrayRef<uint8_t>(bytes).take_front(1))
                  .takeError(),
              "smaller");
  for (LONG length : {0, 65}) {
    fake::writeLength = length;
    expectError(driver->write(bytes).takeError(), "length");
  }
  fake::writeFails = true;
  expectError(driver->write(bytes).takeError(), "OUT");
  fake::finishFails = true;
  fake::complete(7);
  expectError(driver->read(bytes).takeError(), "IN");
  driver->close();

  fake::reset();
  llvm::cantFail(driver->open(config()));
  fake::complete(65);
  expectError(driver->read(bytes).takeError(), "length");
  driver->close();
  fake::reset();
  llvm::cantFail(driver->open(config()));
  fake::complete(7);
  driver->cancel();
  assert(llvm::cantFail(driver->drain(bytes)) == 7 && bytes[6] == 0x5a);
  assert(llvm::cantFail(driver->drain(bytes)) == 0);
  driver->close();
  assert(fake::finishes == 1 && fake::aborts == 0);

  // Both a completed request and an Abort racing a valid completion retain
  // their payload. The same receiver can start again after an ordinary stop.
  fake::reset();
  llvm::cantFail(driver->open(config()));
  fake::abortTail = 9;
  driver->cancel();
  assert(llvm::cantFail(driver->drain(bytes)) == 9 && bytes[8] == 0x5a);
  driver->close();
  fake::reset();
  llvm::cantFail(driver->open(config()));
  driver->cancel();
  assert(llvm::cantFail(driver->drain(bytes)) == 0);
  driver->close();

  for (bool abort : {false, true}) {
    fake::reset();
    llvm::cantFail(driver->open(config()));
    fake::finishFails = true;
    if (!abort)
      fake::complete(7);
    driver->cancel();
    expectError(driver->drain(bytes).takeError(), "IN FinishDataXfer");
    driver->close();
  }

  // If Abort never signals completion, freeing the SDK buffer would be unsafe.
  // The adapter retains it until process exit and refuses to reopen.
  fake::reset();
  llvm::cantFail(driver->open(config()));
  pending = fake::pending;
  const int closes = fake::closes;
  fake::abortStalls = true;
  driver->cancel();
  expectError(driver->drain(bytes).takeError(), "ownership");
  driver->close();
  assert(fake::pending == pending && fake::finishes == 0);
  assert(fake::closes == closes);
  expectError(driver->open(config()), "unretired");
  std::cout
      << "CyAPI transport selection, transfer and lifecycle checks passed\n";
}
