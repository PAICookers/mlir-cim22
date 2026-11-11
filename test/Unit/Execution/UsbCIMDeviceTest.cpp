//===- UsbCIMDeviceTest.cpp - CIM22 USB backend tests --------*- C++ -*-===//

#include "CIM22/Runtime/UsbCIMDevice.h"
#include "CIM22/Runtime/Runtime.h"
#include "CIM22/Runtime/CIMBoardRunner.h"
#include "CIM22/Runtime/CIMPacketCodec.h"
#include "CIM22/Runtime/UsbRecordCodec.h"
#include "Session.h"
#include "UsbDriver.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
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
                         size_t unsolicitedRecords = 0,
                         size_t readChunkLimit = 7)
      : corruptHeader_(corruptHeader), unsolicitedRecords_(unsolicitedRecords),
        readChunkLimit_(readChunkLimit) {}

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
    const size_t count =
        std::min({bytes.size(), incoming_.size(), readChunkLimit_});
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

protected:
  void appendRecord(uint64_t flit, uint8_t channelOffset = 1) {
    const UsbRecord record{
        0, static_cast<uint8_t>((config_.logicalChannel + channelOffset) & 0x7),
        flit};
    auto bytes = UsbRecordCodec::encode(record, config_.wordByteOrder);
    if (corruptHeader_)
      bytes[0] = 0x80;
    incoming_.insert(incoming_.end(), bytes.begin(), bytes.end());
  }

  virtual void appendResponse() {
    appendRecord((uint64_t{5} << 60) | (uint64_t{3} << 18) |
                 (uint64_t{7} << 14) | 6);
    for (uint64_t flit : kResponse)
      appendRecord(flit);
  }

private:
  UsbCIMDeviceConfig config_;
  UsbRecordCodec outgoingCodec_;
  std::mutex mutex_;
  std::condition_variable condition_;
  std::deque<uint8_t> incoming_;
  std::vector<UsbRecord> writes_;
  size_t openCount_ = 0;
  bool corruptHeader_ = false;
  size_t unsolicitedRecords_ = 0;
  size_t readChunkLimit_ = 7;
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
  CIMDeviceReadbackBatch readback;
  readback.targets.push_back({0, 3, 7});
  readback.selection.flits = {0x8000000000000000ULL};
  readback.request.flits = {0x500000000081c000ULL};
  program.readbackBatches.push_back(std::move(readback));
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
  if (error)
    llvm::report_fatal_error(llvm::Twine(llvm::toString(std::move(error))));
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

constexpr uint64_t responseHeader(uint8_t core, uint8_t row) {
  return (uint64_t{5} << 60) | (uint64_t{core} << 18) | (uint64_t{row} << 14) |
         6;
}

class BatchUsbDriver final : public FakeUsbDriver {
public:
  struct Reply {
    uint64_t header;
    uint64_t marker;
    uint8_t bodyChannelOffset = 1;
    uint8_t headChannelOffset = 1;
    bool corruptPadding = false;
  };

  explicit BatchUsbDriver(std::vector<std::vector<Reply>> repliesByRequest,
                          size_t readChunkLimit = 7, bool interleave = false)
      : FakeUsbDriver(false, 0, readChunkLimit),
        repliesByRequest_(std::move(repliesByRequest)),
        interleave_(interleave) {}

private:
  void appendBodyFlit(const Reply &reply, size_t index) {
    const uint64_t marker = index == kResponse.size() - 1
                                ? uint64_t{reply.corruptPadding}
                                : reply.marker;
    appendRecord(kResponse[index] ^ marker, reply.bodyChannelOffset);
  }

  void appendResponse() override {
    assert(requestCount_ < repliesByRequest_.size());
    const auto &replies = repliesByRequest_[requestCount_++];
    for (size_t index = 0; index < replies.size();) {
      const Reply &reply = replies[index++];
      appendRecord(reply.header, reply.headChannelOffset);
      if (interleave_ && index < replies.size()) {
        const Reply &other = replies[index++];
        assert(reply.headChannelOffset != other.headChannelOffset);
        appendRecord(other.header, other.headChannelOffset);
        for (size_t lane = 0; lane < kResponse.size(); ++lane) {
          appendBodyFlit(reply, lane);
          appendBodyFlit(other, lane);
        }
      } else {
        for (size_t lane = 0; lane < kResponse.size(); ++lane)
          appendBodyFlit(reply, lane);
      }
    }
  }

  std::vector<std::vector<Reply>> repliesByRequest_;
  size_t requestCount_ = 0;
  bool interleave_;
};

void testBatchedOutOfOrderReadbacks() {
  CIMDeviceProgram program;
  CIMDeviceReadbackBatch first;
  first.selection.flits = {0x8000000000000000ULL};
  std::vector<std::vector<BatchUsbDriver::Reply>> repliesByRequest(9);
  for (uint8_t row = 0; row < 8; ++row) {
    first.request.flits.push_back(0x5000000000800000ULL |
                                  (uint64_t{row} << 14));
    for (uint8_t core = 0; core < 20; ++core) {
      const size_t outputIndex = row * 20 + core;
      first.targets.push_back({outputIndex, core, row});
      // The fake device releases all 160 replies after all eight requests.
      const uint8_t channelOffset = 1 + core % 2;
      repliesByRequest[7].push_back({responseHeader(core, row), outputIndex,
                                     channelOffset, channelOffset});
    }
  }
  std::reverse(repliesByRequest[7].begin(), repliesByRequest[7].end());
  program.readbackBatches.push_back(std::move(first));
  CIMDeviceReadbackBatch second;
  second.selection.flits = {0x8000000000000001ULL};
  second.request.flits = {0x500000000081c000ULL};
  second.targets.push_back({160, 19, 7});
  repliesByRequest[8].push_back({responseHeader(19, 7), 160});
  program.readbackBatches.push_back(std::move(second));

  auto driver =
      std::make_unique<BatchUsbDriver>(std::move(repliesByRequest), 7, true);
  auto *view = driver.get();
  auto device = detail::createUsbCIMDevice(makeConfig(), std::move(driver));
  expectSuccess(device->open());
  CIMDeviceResponse response;
  expectSuccess(device->execute(program, response));
  assert(response.readbacks.size() == 161);
  for (size_t index = 0; index < response.readbacks.size(); ++index) {
    const size_t outputIndex = index < 160 ? 159 - index : 160;
    assert(response.readbacks[index].outputIndex == outputIndex);
    for (size_t lane = 0; lane + 1 < kResponse.size(); ++lane)
      assert(response.readbacks[index].bodyFlits[lane] ==
             (kResponse[lane] ^ outputIndex));
    assert(response.readbacks[index].bodyFlits.back() == kResponse.back());
  }
  assert(view->writes().size() == 11);
  assert(view->writes()[9].flit == 0x8000000000000001ULL);
  assert(!device->isFaulted());
  device->close();
}

void testBatchedReadbackFaultsStopLaterMacro() {
  using Reply = BatchUsbDriver::Reply;
  const Reply first{responseHeader(3, 7), 0};
  const Reply second{responseHeader(4, 7), 1};
  const std::array cases{
      std::pair{std::vector{first, first}, "duplicate"},
      std::pair{std::vector{first, Reply{responseHeader(5, 7), 1}},
                "unexpected"},
      std::pair{std::vector{first, Reply{second.header ^ 1, 1}},
                "response head"},
      std::pair{std::vector{first, Reply{second.header, 1, 1, 1, true}},
                "non-zero padding"},
      std::pair{std::vector{first, Reply{second.header, 1, 2}},
                "incomplete channel packets"},
      std::pair{std::vector{first}, "timed out"},
      std::pair{std::vector{first, second, second}, "queue overflow"},
  };
  for (const auto &[replies, error] : cases) {
    CIMDeviceProgram program = makeProgram();
    program.readbackBatches.front().targets.push_back({1, 4, 7});
    program.readbackBatches.push_back(program.readbackBatches.front());
    program.readbackBatches.back().selection.flits[0] |= 1;

    auto config = makeConfig();
    config.receiveChunkBytes = 3 * 7 * kUsbRecordBytes;
    config.responseTimeoutMs = 100;
    auto driver = std::make_unique<BatchUsbDriver>(
        std::vector<std::vector<Reply>>{replies}, config.receiveChunkBytes);
    auto *view = driver.get();
    auto device = detail::createUsbCIMDevice(config, std::move(driver));
    expectSuccess(device->open());

    CIMDeviceResponse response;
    assert(takeError(device->execute(program, response)).find(error) !=
           std::string::npos);
    assert(response.readbacks.empty());
    assert(device->isFaulted());
    assert(takeError(device->execute(program, response)).find("faulted") !=
           std::string::npos);
    assert(view->writes().size() == 4);
    device->close();
  }
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
  using Reply = BatchUsbDriver::Reply;
  auto driver = std::make_unique<BatchUsbDriver>(
      std::vector<std::vector<Reply>>{{{responseHeader(3, 7), 0, 1, 1, true}}});
  auto device = detail::createUsbCIMDevice(makeConfig(), std::move(driver));
  expectSuccess(device->open());

  CIMDeviceProgram program = makeProgram();
  program.captureRawReadbacks = true;
  program.readbackBatches.front().selection.flits.front() |= 2;
  CIMDeviceResponse response;
  expectSuccess(device->execute(program, response));
  assert(response.readbacks.size() == 1);
  assert(response.readbacks.front().headerFlit ==
         ((uint64_t{5} << 60) | (uint64_t{3} << 18) | (uint64_t{7} << 14) | 6));
  auto expectedBody = kResponse;
  expectedBody.back() = 1;
  assert(response.readbacks.front().bodyFlits == expectedBody);
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

void testWeightResidencyRejectsPartialWrite() {
  using namespace cim22::execution;
  class PartialWeightDriver final : public FakeUsbDriver {
  public:
    llvm::Expected<size_t> write(llvm::ArrayRef<uint8_t> bytes) override {
      if (fail && !writes().empty()) {
        fail = false;
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "injected partial Weight write");
      }
      return FakeUsbDriver::write(bytes);
    }
    bool fail = true;
  };

  StaticWeightSection section;
  section.words.resize(256);
  CIMWeightCatalog catalog;
  CIMDeviceProgram program;
  program.useWeightResidency = true;
  CIMDeviceOperation write;
  write.weight =
      CIMDeviceWeightWrite{0, catalog.intern(section), BF16Compensation::None};
  program.operations.push_back(std::move(write));

  auto driver = std::make_unique<PartialWeightDriver>();
  auto *view = driver.get();
  auto device = detail::createUsbCIMDevice(makeConfig(), std::move(driver));
  expectSuccess(device->open());
  CIMDeviceResponse response;
  assert(takeError(device->execute(program, response)).find("partial Weight") !=
         std::string::npos);
  assert(view->writes().size() == 1 && device->weightStats().writes == 0);
  assert(takeError(device->execute(program, response)).find("faulted") !=
         std::string::npos);
  expectSuccess(device->open());
  expectSuccess(device->execute(program, response));
  expectSuccess(device->execute(program, response));
  assert(view->writes().size() == 257);
  assert(device->weightStats().writes == 1 && device->weightStats().hits == 1);
  program.operations.front().weight->slot = 1;
  expectSuccess(device->execute(program, response));
  assert(view->writes().size() == 514 && device->weightStats().writes == 2);
  program.operations.front().weight->slot = 0;
  expectSuccess(device->execute(program, response));
  assert(view->writes().size() == 514 && device->weightStats().hits == 2);
  device->close();
}

#ifndef _WIN32
void testBoardWeightResidency(cim22::execution::CIMDataType dataType) {
  using namespace cim22::execution;
  const bool bf16 = dataType == CIMDataType::BF16;
  const std::array<int32_t, 6> route{0, 3, 0, 0, 0, 0};
  auto makeTransaction = [&](int32_t word) {
    StaticWeightSection weight;
    weight.groupId = weight.workId = weight.macroSlot = 0;
    weight.coreSlot = 3;
    weight.route = route;
    weight.words.resize(256);
    weight.words.front() = word;
    weight.exponents.fill(127);
    weight.dataType = dataType;
    return CIMTransaction("cim22-4x5-v1", 1, 1, {{0, {{0, 3, 0, route}}}},
                          {weight}, {}, {{0, 0, 0, 0, dataType}},
                          {{0, 0, 0, 7, route, {}, dataType}}, {});
  };
  const CIMTransaction first = makeTransaction(0);
  const CIMTransaction changed = makeTransaction(1);
  CIMWeightCatalog catalog;
  const std::array firstIdentity{catalog.intern(first.getStaticWeights()[0])};
  const std::array changedIdentity{
      catalog.intern(changed.getStaticWeights()[0])};
  std::array<int8_t, 64> int8Input{};
  std::array<uint16_t, 64> bf16Input{};
  std::array<int32_t, 16> int8Output{};
  std::array<uint16_t, 16> bf16Output{};
  const std::array<CIMInputView, 1> inputs{{{int8Input, bf16Input}}};
  std::array<CIMOutputView, 1> outputViews{{{int8Output, bf16Output}}};
  CIMRunOutputs outputs{outputViews};
  auto driver = std::make_unique<FakeUsbDriver>();
  auto *view = driver.get();
  auto device = detail::createUsbCIMDevice(makeConfig(), std::move(driver));
  CIMBoardRunner runner(*device, {0, 0, 0, 0, 0});
  expectSuccess(device->open());

  auto run = [&](const CIMTransaction &transaction,
                 llvm::ArrayRef<CIMWeightIdentity> identities,
                 BF16Compensation mode = BF16Compensation::None) {
    const size_t begin = view->writes().size();
    expectSuccess(
        runner.run(transaction, {inputs}, outputs, mode, nullptr, identities));
    std::vector<uint64_t> flits;
    for (const auto &record : llvm::ArrayRef(view->writes()).drop_front(begin))
      flits.push_back(record.flit);
    const std::array expectedSuffix{encodeCIMWorkOnceFlit(route),
                                    encodeCIMControlFlit(route, 0, dataType),
                                    encodeCIMOutputCacheReadFlit(route, 7)};
    assert(flits.size() >= expectedSuffix.size());
    assert(std::equal(expectedSuffix.begin(), expectedSuffix.end(),
                      flits.end() - expectedSuffix.size()));
    return flits;
  };
  auto weightHead = [](uint64_t flit) { return (flit >> 60) == 2; };
  auto cold = run(first, firstIdentity);
  const auto coldInt8Output = int8Output;
  const auto coldBF16Output = bf16Output;
  assert(std::count_if(cold.begin(), cold.end(), weightHead) == 1);
  auto weight = std::find_if(cold.begin(), cold.end(), weightHead);
  assert(std::distance(weight, cold.end()) >= 257);
  cold.erase(weight, weight + 257);
  assert(run(first, firstIdentity) == cold);
  assert(int8Output == coldInt8Output && bf16Output == coldBF16Output);

  int8Input.front() = 5;
  bf16Input.front() = 0x3f80;
  const auto newInput = run(first, firstIdentity);
  assert(newInput.size() == cold.size() && newInput != cold);
  assert(std::none_of(newInput.begin(), newInput.end(), weightHead));
  assert(device->weightStats().writes == 1 && device->weightStats().hits == 2);

  const auto overwritten = run(changed, changedIdentity);
  const auto restored = run(first, firstIdentity);
  assert(overwritten.size() == restored.size() && overwritten != restored);
  assert(device->weightStats().writes == 3);
  const BF16Compensation residentMode =
      bf16 ? BF16Compensation::SignSplit : BF16Compensation::None;
  if (bf16) {
    const auto signSplit = run(first, firstIdentity, residentMode);
    assert(std::count_if(signSplit.begin(), signSplit.end(), weightHead) == 1);
    const auto hit = run(first, firstIdentity, residentMode);
    assert(std::none_of(hit.begin(), hit.end(), weightHead));
    assert(bf16Output == (std::array<uint16_t, 16>{}));
  }

  const auto beforeCapture = device->weightStats();
  const size_t beforeCaptureFlits = view->writes().size();
  auto capture = runner.capture(first, {inputs});
  assert(capture && capture->readbacks.size() == 1);
  assert(view->writes().size() - beforeCaptureFlits == restored.size());
  assert(device->weightStats().writes == beforeCapture.writes);
  auto afterCapture = run(first, firstIdentity, residentMode);
  assert(std::count_if(afterCapture.begin(), afterCapture.end(), weightHead) ==
         1);

  device->close();
  expectSuccess(device->open());
  auto afterReopen = run(first, firstIdentity, residentMode);
  assert(std::count_if(afterReopen.begin(), afterReopen.end(), weightHead) ==
         1);
  view->setCorruptHeader(true);
  assert(takeError(runner.run(first, {inputs}, outputs, residentMode, nullptr,
                              firstIdentity))
             .find("reserved header bits") != std::string::npos);
  assert(device->isFaulted());
  view->setCorruptHeader(false);
  expectSuccess(device->open());
  auto afterFault = run(first, firstIdentity, residentMode);
  assert(std::count_if(afterFault.begin(), afterFault.end(), weightHead) == 1);
  device->close();
}

void testPLLReuse(const std::array<uint32_t, 3> &words) {
  using namespace cim22::execution;
  std::vector<CIMGroup> groups;
  std::vector<StaticWeightSection> weights;
  std::vector<DynamicInputBinding> inputs;
  std::vector<ReadbackBinding> readbacks;
  for (int64_t core : {0, 1}) {
    const std::array<int32_t, 6> route{0, static_cast<int32_t>(core), 0, 0, 0,
                                       0};
    CIMGroup group{core, {}};
    for (int64_t macro : {0, 1}) {
      const int64_t work = 2 * core + macro;
      group.works.push_back({work, core, macro, route});
      StaticWeightSection weight;
      weight.groupId = weight.coreSlot = core;
      weight.workId = work;
      weight.macroSlot = macro;
      weight.route = route;
      weight.dataType = CIMDataType::BF16;
      weight.words.resize(256);
      weights.push_back(std::move(weight));
      inputs.push_back({core, work, macro, work, CIMDataType::BF16});
      readbacks.push_back({core, work, macro, 7, route, {}, CIMDataType::BF16});
    }
    groups.push_back(std::move(group));
  }
  const CIMTransaction transaction("cim22-4x5-v1", 1, 1, std::move(groups),
                                   std::move(weights), {}, std::move(inputs),
                                   std::move(readbacks), {});
  std::array<uint16_t, 64> data{};
  std::array<CIMInputView, 4> views;
  views.fill({{}, data});
  std::vector<std::vector<BatchUsbDriver::Reply>> replies;
  // Four transactions. Each Macro has two requests and reuses the same keys
  // as the other Macro. Release the replies in reverse Core order.
  for (size_t run = 0; run < 4; ++run)
    for (size_t macro = 0; macro < 2; ++macro) {
      replies.emplace_back();
      replies.push_back(
          {{responseHeader(1, 7), 2 + macro}, {responseHeader(0, 7), macro}});
    }

  int master = -1, slave = -1;
  char name[128]{};
  assert(::openpty(&master, &slave, name, nullptr, nullptr) == 0);
  std::vector<uint32_t> expected;
  // Reopening must forget the clock; each run follows physical phase changes.
  for (unsigned session = 0; session < 2; ++session) {
    std::optional<uint32_t> current;
    for (unsigned run = 0; run < 2; ++run) {
      for (uint32_t word : words)
        if (current != word) {
          expected.push_back(word);
          current = word;
        }
    }
  }
  std::atomic<size_t> echoed{0};
  std::thread peer([&] {
    for (uint32_t word : expected) {
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
      assert(bytes[0] == (word >> 16) && bytes[1] == ((word >> 8) & 0xff) &&
             bytes[2] == (word & 0xff));
      ++echoed;
      assert(::write(master, bytes.data(), bytes.size()) == 3);
    }
  });
  auto config = makeConfig();
  config.pll = PLLSerialConfig{name, words[0], words[1], words[2]};
  auto device = detail::createUsbCIMDevice(
      config, std::make_unique<BatchUsbDriver>(std::move(replies)));
  CIMBoardRunner runner(*device, {0, 0, 0, 0, 0});
  for (unsigned session = 0; session < 2; ++session) {
    expectSuccess(device->open());
    for (unsigned run = 0; run < 2; ++run) {
      auto response = runner.capture(transaction, {views});
      assert(response && response->readbacks.size() == 4);
      assert((response->pllVerified == std::array<bool, 3>{true, true, true}));
      const std::array<size_t, 4> arrivalOrder{2, 0, 3, 1};
      for (size_t index = 0; index < arrivalOrder.size(); ++index) {
        const auto &readback = response->readbacks[index];
        assert(readback.outputIndex == arrivalOrder[index]);
        assert(readback.bodyFlits.front() ==
               (kResponse.front() ^ arrivalOrder[index]));
      }
    }
    device->close();
  }
  peer.join();
  assert(echoed == expected.size());
  // No additional command may have been sent after the expected prefix.
  pollfd descriptor{master, POLLIN, 0};
  assert(::poll(&descriptor, 1, 0) == 0);
  ::close(master);
  ::close(slave);
}

void testThreePLLPhases(bool wrongWorkEcho, bool disconnectSerial = false) {
  int master = -1, slave = -1;
  char name[128]{};
  assert(::openpty(&master, &slave, name, nullptr, nullptr) == 0);
  char serialDir[] = "/tmp/cim22-pll-XXXXXX";
  std::string port = name;
  if (disconnectSerial) {
    assert(::mkdtemp(serialDir));
    port = std::string(serialDir) + "/serial";
    assert(::symlink(name, port.c_str()) == 0);
  }
  std::atomic<unsigned> echoed{0};
  class PhaseDriver final : public FakeUsbDriver {
  public:
    PhaseDriver(std::atomic<unsigned> &echoed, const char *disconnectPath)
        : echoed_(echoed), disconnectPath_(disconnectPath) {}
    llvm::Expected<size_t> write(llvm::ArrayRef<uint8_t> bytes) override {
      const size_t before = writes().size();
      auto count = FakeUsbDriver::write(bytes);
      if (count && writes().size() != before) {
        const unsigned expected = disconnectPath_ || writes().size() == 1
                                      ? 1
                                      : (writes().size() == 2 ? 2 : 3);
        assert(echoed_.load() == expected);
        if (disconnectPath_ && writes().size() == 1)
          assert(::unlink(disconnectPath_) == 0);
      }
      return count;
    }

  private:
    std::atomic<unsigned> &echoed_;
    const char *disconnectPath_;
  };
  std::thread peer([&] {
    const std::array<uint32_t, 3> words{0x2828a4, 0x2c28a4, 0x3028a4};
    const unsigned phases = disconnectSerial ? 1 : (wrongWorkEcho ? 2 : 3);
    for (unsigned phase = 0; phase < phases; ++phase) {
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
  config.pll = PLLSerialConfig{port, 0x2828a4, 0x2c28a4,
                               disconnectSerial ? 0x2828a4u : 0x3028a4u};
  auto driver = std::make_unique<PhaseDriver>(
      echoed, disconnectSerial ? port.c_str() : nullptr);
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
    assert(takeError(device->execute(makeProgram(), response))
               .find("session is faulted") != std::string::npos);
    assert(view->writes().size() == 1);
    assert(echoed == 2);
  } else {
    expectSuccess(std::move(result));
    assert((response.pllVerified ==
            std::array<bool, 3>{true, !disconnectSerial, !disconnectSerial}));
  }
  device->close();
  ::close(master);
  ::close(slave);
  if (disconnectSerial)
    assert(::rmdir(serialDir) == 0);
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
  testBatchedOutOfOrderReadbacks();
  testBatchedReadbackFaultsStopLaterMacro();
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
  testWeightResidencyRejectsPartialWrite();
#ifndef _WIN32
  testBoardWeightResidency(cim22::execution::CIMDataType::Int8);
  testBoardWeightResidency(cim22::execution::CIMDataType::BF16);
  testThreePLLPhases(false);
  testThreePLLPhases(true);
  testThreePLLPhases(false, true);
  testPLLReuse({0x2828a4, 0x2828a4, 0x2828a4});
  testPLLReuse({0x2828a4, 0x2c28a4, 0x3028a4});
  testPLLReuse({0x2828a4, 0x2828a4, 0x2c28a4});
  testPLLReuse({0x2828a4, 0x2c28a4, 0x2828a4});
  testMissingPLLUsesManualClock();
#endif
  testNativePLLExtensionValidation();
  testReceiveFailureStopsLaterChunks();
  testCloseRetainsLateDataBetweenReads();
  return 0;
}
