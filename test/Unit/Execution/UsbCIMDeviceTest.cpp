//===- UsbCIMDeviceTest.cpp - CIM22 USB backend tests --------*- C++ -*-===//

#include "CIM22/Runtime/UsbCIMDevice.h"
#include "CIM22/Execution/Runtime.h"
#include "CIM22/Runtime/CIMPacketCodec.h"
#include "CIM22/Runtime/UsbRecordCodec.h"
#include "Session.h"
#include "UsbDriver.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#ifndef _WIN32
#include <poll.h>
#include <pty.h>
#endif
#include <string>
#include <thread>
#ifndef _WIN32
#include <unistd.h>
#endif
#include <utility>
#include <vector>

using namespace cim22::runtime;

#ifndef _WIN32
namespace {
std::unique_ptr<detail::UsbDriver> injectedPlatformDriver;
unsigned platformFactoryCalls = 0;
} // namespace

namespace cim22::runtime::detail {
// Interpose the Linux shared-library factory to exercise the real public API
// without opening hardware or adding a production test hook.
LLVM_EXTERNAL_VISIBILITY std::unique_ptr<UsbDriver> createPlatformUsbDriver() {
  ++platformFactoryCalls;
  assert(injectedPlatformDriver);
  return std::move(injectedPlatformDriver);
}
} // namespace cim22::runtime::detail
#endif

namespace {
constexpr std::array<uint64_t, 6> kResponse{
    0x00091fea9580ab55ULL, 0xffff900003f00000ULL, 0x80001fffd6000157ULL,
    0xb72ec2468afffff8ULL, 0x00007ffffc000020ULL, 0x0000000000000000ULL};

class FakeUsbDriver : public detail::UsbDriver {
public:
  explicit FakeUsbDriver(bool corruptHeader = false,
                         size_t unsolicitedRecords = 0)
      : corruptHeader_(corruptHeader), unsolicitedRecords_(unsolicitedRecords) {
  }

  llvm::Error open(const UsbCIMDeviceConfig &config) override {
    config_ = config;
    ++openCount_;
    outgoingCodec_ = UsbRecordCodec(config.wordByteOrder);
    incoming_.clear();
    writes_.clear();
    open_ = true;
    cancelled_ = false;
    for (size_t index = 0; index < unsolicitedRecords_; ++index)
      appendRecord(static_cast<uint64_t>(index));
    return llvm::Error::success();
  }

  llvm::Expected<size_t> write(llvm::ArrayRef<uint8_t> bytes) override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!open_)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "fake USB driver is closed");
    if (!readStarted_)
      writeBeforeRead_ = true;
    const size_t count = std::min<size_t>(5, bytes.size());
    llvm::SmallVector<UsbRecord, 2> decoded;
    if (llvm::Error error =
            outgoingCodec_.append(bytes.take_front(count), decoded))
      return std::move(error);
    for (const UsbRecord &record : decoded) {
      writes_.push_back(record);
      if (((record.flit >> 60) & 0xf) == 5 && ((record.flit >> 23) & 1) != 0)
        appendResponse();
    }
    condition_.notify_all();
    return count;
  }

  llvm::Expected<size_t> read(llvm::MutableArrayRef<uint8_t> bytes) override {
    std::unique_lock<std::mutex> lock(mutex_);
    readStarted_ = true;
    condition_.notify_all();
    condition_.wait_for(lock, std::chrono::milliseconds(20),
                        [&] { return !incoming_.empty() || cancelled_; });
    if (incoming_.empty())
      return size_t{0};
    const size_t count = std::min({bytes.size(), incoming_.size(), size_t{7}});
    for (size_t index = 0; index < count; ++index) {
      bytes[index] = incoming_.front();
      incoming_.pop_front();
    }
    return count;
  }

  void cancel() override {
    std::lock_guard<std::mutex> lock(mutex_);
    cancelled_ = true;
    condition_.notify_all();
  }

  void close() override {
    std::lock_guard<std::mutex> lock(mutex_);
    open_ = false;
    cancelled_ = true;
    condition_.notify_all();
  }

  const std::vector<UsbRecord> &writes() const { return writes_; }
  bool cancelled() const { return cancelled_; }
  size_t openCount() const { return openCount_; }
  size_t pendingWriteBytes() const { return outgoingCodec_.pendingBytes(); }
  void setCorruptHeader(bool value) { corruptHeader_ = value; }
  bool writeBeforeRead() {
    std::lock_guard<std::mutex> lock(mutex_);
    return writeBeforeRead_;
  }

private:
  void appendRecord(uint64_t flit) {
    const UsbRecord record{
        0, static_cast<uint8_t>((config_.logicalChannel + 1) & 0x7), flit};
    auto bytes = UsbRecordCodec::encode(record, config_.wordByteOrder);
    if (corruptHeader_)
      bytes[0] = 0x80;
    incoming_.insert(incoming_.end(), bytes.begin(), bytes.end());
  }

  void appendResponse() {
    appendRecord((uint64_t{5} << 60) | (uint64_t{3} << 18) |
                 (uint64_t{7} << 14) | 6);
    for (uint64_t flit : kResponse)
      appendRecord(flit);
  }

  UsbCIMDeviceConfig config_;
  UsbRecordCodec outgoingCodec_;
  std::mutex mutex_;
  std::condition_variable condition_;
  std::deque<uint8_t> incoming_;
  std::vector<UsbRecord> writes_;
  size_t openCount_ = 0;
  bool corruptHeader_ = false;
  size_t unsolicitedRecords_ = 0;
  bool readStarted_ = false;
  bool writeBeforeRead_ = false;
  bool open_ = false;
  bool cancelled_ = false;
};

UsbCIMDeviceConfig makeConfig() {
  UsbCIMDeviceConfig config;
  config.vendorId = 0x04b4;
  config.productId = 0x00f1;
  config.bulkInEndpoint = 0x81;
  config.bulkOutEndpoint = 0x01;
  config.physicalLink = 1;
  config.logicalChannel = 2;
  config.startupDelayMs = 1;
  config.responseTimeoutMs = 1000;
  config.receiveChunkBytes = 31;
  config.maxTransferBytes = 17;
  return config;
}

CIMDeviceProgram makeProgram() {
  CIMDeviceProgram program;
  CIMDeviceOperation select;
  select.flits = {0x8000000000000000ULL};
  program.operations.push_back(std::move(select));
  CIMDeviceOperation once;
  once.flits = {0x9000000000000001ULL};
  once.phase = CIMDevicePhase::Work;
  program.operations.push_back(std::move(once));
  CIMDeviceReadbackCommand readback;
  readback.outputIndex = 0;
  readback.selection.flits = {0x8000000000000000ULL};
  readback.request.flits = {0x500000000081c000ULL};
  program.readbacks.push_back(std::move(readback));
  return program;
}

void testSupplierInteractionDefaults() {
  UsbCIMDeviceConfig config;
  assert(config.wordByteOrder == UsbWordByteOrder::BigEndian);
  assert(config.startupDelayMs == 20);
  assert(config.maxTransferBytes == 512);
  assert(config.receiveChunkBytes == 32768);
  assert(config.responseTimeoutMs == 7000);
  assert(config.physicalLink == 0 && config.logicalChannel == 3);
  const auto record = UsbRecordCodec::encode(
      {config.physicalLink, config.logicalChannel, 0xa00000000003f000ULL});
  assert(record[0] == 0 && record[1] == 0 && record[2] == 0 && record[3] == 3);
}

void expectSuccess(llvm::Error error) {
  assert(!error && "unexpected LLVM error");
}

std::string takeError(llvm::Error error) {
  assert(error && "expected LLVM error");
  return llvm::toString(std::move(error));
}

void testRecordCodecResidualBytes() {
  UsbRecordCodec codec;
  const UsbRecord expected{2, 5, 0x123456789abcdef0ULL};
  const auto encoded = UsbRecordCodec::encode(expected);
  const UsbRecord secondExpected{1, 4, 0x0fedcba987654321ULL};
  const auto secondEncoded = UsbRecordCodec::encode(secondExpected);
  llvm::SmallVector<UsbRecord, 1> records;
  expectSuccess(
      codec.append(llvm::ArrayRef<uint8_t>(encoded).take_front(5), records));
  assert(records.empty() && codec.pendingBytes() == 5);
  expectSuccess(
      codec.append(llvm::ArrayRef<uint8_t>(encoded).drop_front(5), records));
  assert(records.size() == 1 && records.front().physicalLink == 2 &&
         records.front().logicalChannel == 5 &&
         records.front().flit == expected.flit);
  llvm::SmallVector<uint8_t, 2 * kUsbRecordBytes> coalesced;
  coalesced.append(encoded.begin(), encoded.end());
  coalesced.append(secondEncoded.begin(), secondEncoded.end());
  records.clear();
  expectSuccess(codec.append(coalesced, records));
  assert(records.size() == 2 && records[0].flit == expected.flit &&
         records[1].flit == secondExpected.flit);
}

void testRecordCodecWordByteOrder() {
  // Supplier left_down_1 return field (0x3f) in a big-endian USB record.
  const std::array<int32_t, 6> route{};
  const std::array<int32_t, 3> testCore{-31, 0, 0};
  const uint64_t returnFlit = encodeCIMReturnRouteFlit(route, 0, testCore);
  assert(returnFlit == 0xa00000000003f000ULL);
  const std::array<uint8_t, kUsbRecordBytes> supplierReturn{
      0x00, 0x00, 0x00, 0x13, 0xa0, 0x00, 0x00, 0x00, 0x00, 0x03, 0xf0, 0x00};
  assert(UsbRecordCodec::encode({2, 3, returnFlit}) == supplierReturn);

  const UsbRecord record{2, 3, 0x1122334455667788ULL};
  const std::array<uint8_t, kUsbRecordBytes> bigEndian{
      0x00, 0x00, 0x00, 0x13, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88};
  const std::array<uint8_t, kUsbRecordBytes> littleEndian{
      0x13, 0x00, 0x00, 0x00, 0x44, 0x33, 0x22, 0x11, 0x88, 0x77, 0x66, 0x55};
  assert(UsbRecordCodec::encode(record, UsbWordByteOrder::BigEndian) ==
         bigEndian);
  assert(UsbRecordCodec::encode(record, UsbWordByteOrder::LittleEndian) ==
         littleEndian);

  for (const auto &[order, bytes] :
       {std::pair{UsbWordByteOrder::BigEndian, bigEndian},
        std::pair{UsbWordByteOrder::LittleEndian, littleEndian}}) {
    UsbRecordCodec codec(order);
    llvm::SmallVector<UsbRecord, 1> decoded;
    expectSuccess(codec.append(bytes, decoded));
    assert(decoded.size() == 1 && decoded.front().physicalLink == 2 &&
           decoded.front().logicalChannel == 3 &&
           decoded.front().flit == record.flit);
  }
}

void testPartialTransfersAndIndependentReturnSource() {
  auto driver = std::make_unique<FakeUsbDriver>();
  FakeUsbDriver *driverView = driver.get();
  auto device = detail::createUsbCIMDevice(makeConfig(), std::move(driver));
  expectSuccess(device->open());

  CIMDeviceResponse response;
  expectSuccess(device->execute(makeProgram(), response));
  assert(response.readbacks.size() == 1);
  assert(response.readbacks[0].headerFlit ==
         ((uint64_t{5} << 60) | (uint64_t{3} << 18) | (uint64_t{7} << 14) | 6));
  assert(response.readbacks[0].bodyFlits == kResponse);
  assert(driverView->writes().size() == 4);
  assert(driverView->writes()[0].flit == 0x8000000000000000ULL);
  assert(driverView->writes()[1].flit == 0x9000000000000001ULL);
  assert(driverView->writes().back().flit == 0x500000000081c000ULL);
  assert(!driverView->writeBeforeRead());
  device->close();
}

void testLittleEndianDeviceReadback() {
  UsbCIMDeviceConfig config = makeConfig();
  config.wordByteOrder = UsbWordByteOrder::LittleEndian;
  auto driver = std::make_unique<FakeUsbDriver>();
  auto device = detail::createUsbCIMDevice(config, std::move(driver));
  expectSuccess(device->open());

  CIMDeviceResponse response;
  expectSuccess(device->execute(makeProgram(), response));
  assert(response.readbacks.size() == 1);
  assert(response.readbacks.front().bodyFlits == kResponse);
  device->close();
}

void testStickyFault() {
  auto driver = std::make_unique<FakeUsbDriver>(true);
  auto device = detail::createUsbCIMDevice(makeConfig(), std::move(driver));
  expectSuccess(device->open());

  CIMDeviceResponse response;
  std::string first = takeError(device->execute(makeProgram(), response));
  assert(first.find("reserved header bits") != std::string::npos);
  std::string second = takeError(device->execute(makeProgram(), response));
  assert(second.find("session is faulted") != std::string::npos);
  device->close();
}

void testTimeoutAndCancellation() {
  class SilentUsbDriver final : public FakeUsbDriver {
  public:
    llvm::Expected<size_t> write(llvm::ArrayRef<uint8_t> bytes) override {
      return bytes.size();
    }
  };

  UsbCIMDeviceConfig config = makeConfig();
  config.responseTimeoutMs = 30;
  auto driver = std::make_unique<SilentUsbDriver>();
  SilentUsbDriver *driverView = driver.get();
  auto device = detail::createUsbCIMDevice(config, std::move(driver));
  expectSuccess(device->open());

  CIMDeviceResponse response;
  std::string message = takeError(device->execute(makeProgram(), response));
  assert(message.find("timed out") != std::string::npos);
  device->close();
  assert(driverView->cancelled());
}

void testWriteFailureIsSticky() {
  class FailingUsbDriver final : public FakeUsbDriver {
  public:
    llvm::Expected<size_t> write(llvm::ArrayRef<uint8_t> bytes) override {
      if (writes_++ == 0)
        return FakeUsbDriver::write(bytes);
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "injected USB write failure");
    }

  private:
    size_t writes_ = 0;
  };

  auto driver = std::make_unique<FailingUsbDriver>();
  auto *driverView = driver.get();
  auto device = detail::createUsbCIMDevice(makeConfig(), std::move(driver));
  expectSuccess(device->open());
  CIMDeviceResponse response;
  const std::string first = takeError(device->execute(makeProgram(), response));
  assert(first.find("injected USB write failure") != std::string::npos);
  const std::string second =
      takeError(device->execute(makeProgram(), response));
  assert(second.find("session is faulted") != std::string::npos);
  assert(driverView->pendingWriteBytes() == 5);
  device->close();
  assert(device->isFaulted());
  expectSuccess(device->open());
  assert(driverView->openCount() == 2);
  assert(driverView->pendingWriteBytes() == 0);
}

#ifndef _WIN32
void testFailedOpenPreservesSessionFault() {
  class EarlyFailureDriver final : public FakeUsbDriver {
  public:
    llvm::Expected<size_t> read(llvm::MutableArrayRef<uint8_t>) override {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "injected startup IN failure");
    }
  };
  injectedPlatformDriver = std::make_unique<EarlyFailureDriver>();
  Session session;
  session.model.backend = RuntimeBackend::USB;
  CIM22UsbConfig config{};
  config.vendor_id = 0x1234;
  config.product_id = 0x5678;
  config.bus_number = config.device_address = 0xff;
  config.bulk_in_endpoint = 0x81;
  config.bulk_out_endpoint = 0x01;
  config.logical_channel = 3;
  config.transfer_timeout_ms = 1000;
  config.receive_poll_timeout_ms = 200;
  config.response_timeout_ms = 7000;
  std::array<char, 256> error{};
  assert(cim22_session_configure_usb(&session, &config, error.data(),
                                     error.size()) != 0);
  assert(std::string(error.data()).find("startup IN failure") !=
         std::string::npos);
  assert(session.device && session.device->isFaulted());
  assert(!session.runner);
  CIM22PLLConfig pll{};
  assert(cim22_session_configure_usb_with_pll(&session, &config, &pll,
                                              error.data(), error.size()) != 0);
  assert(std::string(error.data()).find("failed before reconfiguration") !=
         std::string::npos);
  assert(!session.device && !session.runner);
  injectedPlatformDriver = std::make_unique<FakeUsbDriver>();
  assert(cim22_session_configure_usb(&session, &config, error.data(),
                                     error.size()) == 0);
  assert(session.device && session.runner);
  assert(platformFactoryCalls == 2);
}

#endif

void testExplicitReopenAfterFault() {
  auto driver = std::make_unique<FakeUsbDriver>(true);
  FakeUsbDriver *driverView = driver.get();
  auto device = detail::createUsbCIMDevice(makeConfig(), std::move(driver));
  expectSuccess(device->open());
  CIMDeviceResponse response;
  assert(takeError(device->execute(makeProgram(), response))
             .find("reserved header bits") != std::string::npos);
  device->close();

  driverView->setCorruptHeader(false);
  assert(device->isFaulted());
  expectSuccess(device->open());
  assert(driverView->openCount() == 2);
  expectSuccess(device->execute(makeProgram(), response));
  device->close();

  // Reconfiguration reports the previous failure once, releases that device,
  // and leaves this Session available for an explicit later retry.
  driverView->setCorruptHeader(true);
  expectSuccess(device->open());
  llvm::consumeError(device->execute(makeProgram(), response));
  Session session;
  session.model.backend = RuntimeBackend::USB;
  session.device = std::move(device);
  CIM22UsbConfig config{};
  std::array<char, 256> error{};
  assert(cim22_session_configure_usb(&session, &config, error.data(),
                                     error.size()) != 0);
  assert(std::string(error.data()).find("USB receive/transaction failed") !=
         std::string::npos);
  assert(!session.device && !session.runner);
}

void testRejectsUnsolicitedAndOverflowResponses() {
  class BurstDriver final : public FakeUsbDriver {
  public:
    explicit BurstDriver(size_t count)
        : FakeUsbDriver(false, count), remaining_(count * kUsbRecordBytes) {}
    llvm::Expected<size_t> read(llvm::MutableArrayRef<uint8_t> bytes) override {
      if (!remaining_)
        return FakeUsbDriver::read(bytes);
      size_t total = 0;
      while (total < remaining_) {
        auto part = FakeUsbDriver::read(bytes.slice(total, remaining_ - total));
        if (!part)
          return part.takeError();
        assert(*part > 0);
        total += *part;
      }
      remaining_ = 0;
      return total;
    }

  private:
    size_t remaining_;
  };
  for (size_t count : {size_t{1}, size_t{8}}) {
    auto config = makeConfig();
    config.receiveChunkBytes = 8 * kUsbRecordBytes;
    config.startupDelayMs = 0;
    auto device = detail::createUsbCIMDevice(
        config, std::make_unique<BurstDriver>(count));
    llvm::Error openError = device->open();
    if (openError) {
      assert(count == 8);
      assert(llvm::toString(std::move(openError)).find("queue overflow") !=
             std::string::npos);
      continue;
    }
    // Wait for the whole burst, not just receive-thread startup.
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (!device->isFaulted() && std::chrono::steady_clock::now() < deadline)
      std::this_thread::yield();
    assert(device->isFaulted());
    CIMDeviceResponse response;
    const std::string message =
        takeError(device->execute(makeProgram(), response));
    assert(message.find(count == 1 ? "stale response" : "queue overflow") !=
           std::string::npos);
    device->close();
  }
}

void testRejectsInvalidTransferConfig() {
  UsbCIMDeviceConfig config = makeConfig();
  config.maxTransferBytes = kUsbRecordBytes * 256 + 1;
  auto device =
      detail::createUsbCIMDevice(config, std::make_unique<FakeUsbDriver>());
  assert(takeError(device->open()).find("transfer sizes") != std::string::npos);
}

void testRejectsInvalidEndpointAddresses() {
  for (const auto &[input, output] : {std::pair<uint8_t, uint8_t>{0x01, 0x01},
                                      {0x81, 0x81},
                                      {0x80, 0x01},
                                      {0x81, 0x00},
                                      {0x91, 0x01},
                                      {0x81, 0x11}}) {
    UsbCIMDeviceConfig config = makeConfig();
    config.bulkInEndpoint = input;
    config.bulkOutEndpoint = output;
    auto device =
        detail::createUsbCIMDevice(config, std::make_unique<FakeUsbDriver>());
    assert(takeError(device->open()).find("endpoint addresses") !=
           std::string::npos);
  }
}

void testRejectsInvalidWordByteOrder() {
  UsbCIMDeviceConfig config = makeConfig();
  config.wordByteOrder = static_cast<UsbWordByteOrder>(2);
  auto device =
      detail::createUsbCIMDevice(config, std::make_unique<FakeUsbDriver>());
  assert(takeError(device->open()).find("word byte order") !=
         std::string::npos);
}

void testPreservesRawBF16Readback() {
  auto driver = std::make_unique<FakeUsbDriver>();
  auto device = detail::createUsbCIMDevice(makeConfig(), std::move(driver));
  expectSuccess(device->open());

  CIMDeviceProgram program = makeProgram();
  program.readbacks.front().dataType = cim22::execution::CIMDataType::BF16;
  CIMDeviceResponse response;
  expectSuccess(device->execute(program, response));
  assert(response.readbacks.size() == 1);
  assert(response.readbacks.front().headerFlit ==
         ((uint64_t{5} << 60) | (uint64_t{3} << 18) | (uint64_t{7} << 14) | 6));
  assert(response.readbacks.front().bodyFlits == kResponse);
  device->close();
}

void testPreservesErrorDuringStop() {
  class CancelErrorUsbDriver final : public FakeUsbDriver {
  public:
    llvm::Error open(const UsbCIMDeviceConfig &config) override {
      std::lock_guard<std::mutex> lock(mutex_);
      cancelled_ = false;
      return FakeUsbDriver::open(config);
    }

    llvm::Expected<size_t> read(llvm::MutableArrayRef<uint8_t>) override {
      std::unique_lock<std::mutex> lock(mutex_);
      reading_ = true;
      condition_.notify_all();
      condition_.wait(lock, [&] { return cancelled_; });
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "injected USB receive failure during stop");
    }

    void cancel() override {
      std::lock_guard<std::mutex> lock(mutex_);
      cancelled_ = true;
      condition_.notify_all();
    }

    void waitUntilReading() {
      std::unique_lock<std::mutex> lock(mutex_);
      assert(condition_.wait_for(lock, std::chrono::seconds(1),
                                 [&] { return reading_; }));
    }

  private:
    std::mutex mutex_;
    std::condition_variable condition_;
    bool cancelled_ = false;
    bool reading_ = false;
  };

  auto driver = std::make_unique<CancelErrorUsbDriver>();
  auto *view = driver.get();
  auto device = detail::createUsbCIMDevice(makeConfig(), std::move(driver));
  expectSuccess(device->open());
  view->waitUntilReading();
  device->close();
  assert(device->isFaulted());
}

void testDrainsBeforeClose() {
  class TailUsbDriver final : public FakeUsbDriver {
  public:
    explicit TailUsbDriver(bool fail) : fail_(fail) {}
    llvm::Expected<size_t>
    drain(llvm::MutableArrayRef<uint8_t> bytes) override {
      if (drained_)
        return size_t{0};
      drained_ = true;
      if (fail_)
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "injected tail failure");
      auto record = UsbRecordCodec::encode(UsbRecord{0, 0, 123});
      std::copy(record.begin(), record.end(), bytes.begin());
      return record.size();
    }
    bool drained_ = false;

  private:
    bool fail_;
  };
  for (bool fail : {false, true}) {
    auto driver = std::make_unique<TailUsbDriver>(fail);
    auto *view = driver.get();
    auto device = detail::createUsbCIMDevice(makeConfig(), std::move(driver));
    expectSuccess(device->open());
    device->close();
    assert(view->drained_);
    // A successful but unsolicited tail must remain visible to the final
    // transaction check, just as a late receive failure must.
    assert(device->isFaulted());
  }
}
void testSupplierWeightPacketChunks() {
  class WholeTransferDriver final : public FakeUsbDriver {
  public:
    llvm::Expected<size_t> write(llvm::ArrayRef<uint8_t> bytes) override {
      sizes.push_back(bytes.size());
      return bytes.size();
    }
    std::vector<size_t> sizes;
  };
  auto config = makeConfig();
  config.maxTransferBytes = 512;
  auto driver = std::make_unique<WholeTransferDriver>();
  auto *view = driver.get();
  auto device = detail::createUsbCIMDevice(config, std::move(driver));
  expectSuccess(device->open());
  CIMDeviceProgram program;
  CIMDeviceOperation weights;
  weights.flits.resize(257);
  program.operations.push_back(std::move(weights));
  CIMDeviceResponse response;
  expectSuccess(device->execute(program, response));
  assert(
      (view->sizes == std::vector<size_t>{512, 512, 512, 512, 512, 512, 12}));
  assert((response.pllVerified == std::array<bool, 3>{false, false, false}));
  device->close();
}

#ifndef _WIN32
void testThreePLLPhases(bool wrongWorkEcho) {
  int master = -1, slave = -1;
  char name[128]{};
  assert(::openpty(&master, &slave, name, nullptr, nullptr) == 0);
  std::atomic<unsigned> echoed{0};
  class PhaseDriver final : public FakeUsbDriver {
  public:
    explicit PhaseDriver(std::atomic<unsigned> &echoed) : echoed_(echoed) {}
    llvm::Expected<size_t> write(llvm::ArrayRef<uint8_t> bytes) override {
      const size_t before = writes().size();
      auto count = FakeUsbDriver::write(bytes);
      if (count && writes().size() != before) {
        const unsigned expected =
            writes().size() == 1 ? 1 : (writes().size() == 2 ? 2 : 3);
        assert(echoed_.load() == expected);
      }
      return count;
    }

  private:
    std::atomic<unsigned> &echoed_;
  };
  std::thread peer([&] {
    const std::array<uint32_t, 3> words{0x2828a4, 0x2c28a4, 0x3028a4};
    for (unsigned phase = 0; phase < (wrongWorkEcho ? 2u : 3u); ++phase) {
      std::array<uint8_t, 3> bytes{};
      size_t offset = 0;
      while (offset < bytes.size()) {
        pollfd descriptor{master, POLLIN, 0};
        assert(::poll(&descriptor, 1, 1000) == 1);
        const ssize_t count =
            ::read(master, bytes.data() + offset, bytes.size() - offset);
        assert(count > 0);
        offset += count;
      }
      assert(bytes[0] == (words[phase] >> 16) &&
             bytes[1] == ((words[phase] >> 8) & 0xff) &&
             bytes[2] == (words[phase] & 0xff));
      if (wrongWorkEcho && phase == 1)
        bytes.back() ^= 1;
      echoed.store(phase + 1);
      assert(::write(master, bytes.data(), bytes.size()) == 3);
    }
  });
  auto config = makeConfig();
  config.pll = PLLSerialConfig{name, 0x2828a4, 0x2c28a4, 0x3028a4};
  auto driver = std::make_unique<PhaseDriver>(echoed);
  auto *view = driver.get();
  auto device = detail::createUsbCIMDevice(config, std::move(driver));
  expectSuccess(device->open());
  CIMDeviceResponse response;
  llvm::Error result = device->execute(makeProgram(), response);
  peer.join();
  if (wrongWorkEcho) {
    assert(takeError(std::move(result)).find("PLL echo timed out") !=
           std::string::npos);
    assert(view->writes().size() == 1);
    assert(device->isFaulted());
  } else {
    expectSuccess(std::move(result));
    assert((response.pllVerified == std::array<bool, 3>{true, true, true}));
  }
  device->close();
  ::close(master);
  ::close(slave);
}

void testMissingPLLUsesManualClock() {
  auto config = makeConfig();
  config.pll = PLLSerialConfig{"/missing-cim22-pll-device", 0x2828a4, 0x2828a4,
                               0x2828a4};
  auto device =
      detail::createUsbCIMDevice(config, std::make_unique<FakeUsbDriver>());
  expectSuccess(device->open());
  CIMDeviceResponse response;
  expectSuccess(device->execute(makeProgram(), response));
  assert((response.pllVerified == std::array<bool, 3>{false, false, false}));
  device->close();
}

#endif

void testNativePLLExtensionValidation() {
  Session session;
  session.model.backend = RuntimeBackend::USB;
  CIM22UsbConfig config{};
  CIM22PLLConfig pll{"/missing-cim22-pll-device", 0x1000000, 0x2828a4,
                     0x2828a4};
  std::array<char, 256> error{};
  assert(cim22_session_configure_usb_with_pll(&session, &config, &pll,
                                              error.data(), error.size()) != 0);
  assert(std::string(error.data()).find("three bytes") != std::string::npos);
  assert(!session.device);
}

struct ReadFailureState {
  std::mutex mutex;
  std::condition_variable condition;
  bool requested = false;
  bool consumed = false;
  bool cancelled = false;
};

class PublishedReadError final : public llvm::ErrorInfo<PublishedReadError> {
public:
  static char ID;
  explicit PublishedReadError(ReadFailureState &state) : state_(state) {}
  void log(llvm::raw_ostream &stream) const override {
    stream << "injected asynchronous IN error";
    // Portable code holds its receive lock while consuming this error. Once
    // write resumes, its next chunk must see the published receive fault.
    std::lock_guard<std::mutex> lock(state_.mutex);
    state_.consumed = true;
    state_.condition.notify_all();
  }
  std::error_code convertToErrorCode() const override {
    return llvm::inconvertibleErrorCode();
  }

private:
  ReadFailureState &state_;
};
char PublishedReadError::ID = 0;

void testReceiveFailureStopsLaterChunks() {
  class FailureDriver final : public FakeUsbDriver {
  public:
    llvm::Error open(const UsbCIMDeviceConfig &config) override {
      std::lock_guard<std::mutex> lock(state.mutex);
      state.requested = state.consumed = state.cancelled = false;
      return FakeUsbDriver::open(config);
    }
    llvm::Expected<size_t> write(llvm::ArrayRef<uint8_t> bytes) override {
      ++calls;
      std::unique_lock<std::mutex> lock(state.mutex);
      state.requested = true;
      state.condition.notify_all();
      assert(state.condition.wait_for(lock, std::chrono::seconds(1),
                                      [&] { return state.consumed; }));
      return bytes.size();
    }
    llvm::Expected<size_t> read(llvm::MutableArrayRef<uint8_t>) override {
      std::unique_lock<std::mutex> lock(state.mutex);
      state.condition.wait(lock,
                           [&] { return state.requested || state.cancelled; });
      if (state.cancelled)
        return size_t{0};
      return llvm::make_error<PublishedReadError>(state);
    }
    void cancel() override {
      std::lock_guard<std::mutex> lock(state.mutex);
      state.cancelled = true;
      state.condition.notify_all();
    }
    size_t calls = 0;
    ReadFailureState state;
  };
  auto config = makeConfig();
  config.maxTransferBytes = 512;
  auto driver = std::make_unique<FailureDriver>();
  auto *view = driver.get();
  auto device = detail::createUsbCIMDevice(config, std::move(driver));
  expectSuccess(device->open());
  CIMDeviceProgram program = makeProgram();
  program.operations.front().flits.resize(257);
  CIMDeviceResponse response;
  assert(
      takeError(device->execute(program, response)).find("asynchronous IN") !=
      std::string::npos);
  assert(view->calls == 1);
  assert(device->isFaulted());
  device->close();
}

void testCloseRetainsLateDataBetweenReads() {
  class LateDriver final : public FakeUsbDriver {
  public:
    llvm::Error open(const UsbCIMDeviceConfig &config) override {
      std::lock_guard<std::mutex> lock(mutex);
      cancelled = false;
      calls = 0;
      return FakeUsbDriver::open(config);
    }
    llvm::Expected<size_t> read(llvm::MutableArrayRef<uint8_t> bytes) override {
      std::unique_lock<std::mutex> lock(mutex);
      ++calls;
      condition.notify_all();
      if (calls == 1) {
        condition.wait(lock, [&] { return cancelled; });
        return size_t{0};
      }
      if (calls > 2)
        return size_t{0};
      bytes[0] = 0;
      return size_t{1};
    }
    llvm::Expected<size_t>
    drain(llvm::MutableArrayRef<uint8_t> bytes) override {
      return read(bytes);
    }
    void cancel() override {
      std::lock_guard<std::mutex> lock(mutex);
      cancelled = true;
      condition.notify_all();
    }
    std::mutex mutex;
    std::condition_variable condition;
    bool cancelled = false;
    unsigned calls = 0;
  };
  auto driver = std::make_unique<LateDriver>();
  auto *view = driver.get();
  auto device = detail::createUsbCIMDevice(makeConfig(), std::move(driver));
  expectSuccess(device->open());
  {
    std::unique_lock<std::mutex> lock(view->mutex);
    assert(view->condition.wait_for(lock, std::chrono::seconds(1),
                                    [&] { return view->calls != 0; }));
  }
  device->close();
  assert(device->isFaulted());
  expectSuccess(device->open());
  assert(!device->isFaulted());
}
} // namespace

int main() {
  testSupplierInteractionDefaults();
  testRecordCodecResidualBytes();
  testRecordCodecWordByteOrder();
  testPartialTransfersAndIndependentReturnSource();
  testLittleEndianDeviceReadback();
  testStickyFault();
  testTimeoutAndCancellation();
  testWriteFailureIsSticky();
  testExplicitReopenAfterFault();
#ifndef _WIN32
  testFailedOpenPreservesSessionFault();
#endif
  testRejectsUnsolicitedAndOverflowResponses();
  testRejectsInvalidTransferConfig();
  testRejectsInvalidEndpointAddresses();
  testRejectsInvalidWordByteOrder();
  testPreservesRawBF16Readback();
  testPreservesErrorDuringStop();
  testDrainsBeforeClose();
  testSupplierWeightPacketChunks();
#ifndef _WIN32
  testThreePLLPhases(false);
  testThreePLLPhases(true);
  testMissingPLLUsesManualClock();
#endif
  testNativePLLExtensionValidation();
  testReceiveFailureStopsLaterChunks();
  testCloseRetainsLateDataBetweenReads();
  return 0;
}
