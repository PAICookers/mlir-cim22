//===- UsbCIMDeviceTest.cpp - CIM22 USB backend tests --------*- C++ -*-===//

#include "CIM22/Runtime/UsbCIMDevice.h"
#include "CIM22/Execution/Runtime.h"
#include "CIM22/Runtime/Detail/UsbDriver.h"
#include "CIM22/Runtime/UsbRecordCodec.h"
#include "Session.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

using namespace cim22::runtime;

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
    if (openCount_++ == 0)
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
        static_cast<uint8_t>((config_.physicalLink + 1) & 0x3),
        static_cast<uint8_t>((config_.logicalChannel + 1) & 0x7), flit};
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
  program.operations.push_back(std::move(once));
  CIMDeviceReadbackCommand readback;
  readback.outputIndex = 0;
  readback.request.flits = {0x8000000000000000ULL, 0x500000000081c000ULL};
  program.readbacks.push_back(std::move(readback));
  return program;
}

void testSupplierInteractionDefaults() {
  UsbCIMDeviceConfig config;
  assert(config.wordByteOrder == UsbWordByteOrder::BigEndian);
  assert(config.startupDelayMs == 20);
  assert(config.maxTransferBytes == kUsbRecordBytes * 256);
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
  assert(takeError(device->open()).find("remains faulted") !=
         std::string::npos);
  assert(driverView->openCount() == 1);
  assert(driverView->pendingWriteBytes() == 5);
}

void testReopenPreservesSessionFault() {
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
  assert(takeError(device->open()).find("remains faulted") !=
         std::string::npos);
  assert(driverView->openCount() == 1);
  device->close();

  Session session;
  session.model.backend = RuntimeBackend::USB;
  auto *faultedDevice = device.get();
  session.device = std::move(device);
  CIM22UsbConfig config{};
  std::array<char, 256> error{};
  assert(cim22_session_configure_usb(&session, &config, error.data(),
                                     error.size()) != 0);
  assert(std::string(error.data()).find("USB session is faulted") !=
         std::string::npos);
  assert(session.device.get() == faultedDevice);
  assert(driverView->openCount() == 1);
}

void testRejectsUnsolicitedAndOverflowResponses() {
  for (size_t count : {size_t{1}, size_t{8}}) {
    auto device = detail::createUsbCIMDevice(
        makeConfig(), std::make_unique<FakeUsbDriver>(false, count));
    llvm::Error openError = device->open();
    if (count == 8) {
      assert(openError);
      assert(llvm::toString(std::move(openError)).find("queue overflow") !=
             std::string::npos);
      continue;
    }
    expectSuccess(std::move(openError));
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

void testConsumesCancelledReadError() {
  class CancelErrorUsbDriver final : public FakeUsbDriver {
  public:
    llvm::Error open(const UsbCIMDeviceConfig &config) override {
      std::lock_guard<std::mutex> lock(mutex_);
      cancelled_ = false;
      return FakeUsbDriver::open(config);
    }

    llvm::Expected<size_t> read(llvm::MutableArrayRef<uint8_t>) override {
      std::unique_lock<std::mutex> lock(mutex_);
      condition_.wait(lock, [&] { return cancelled_; });
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "injected USB cancellation");
    }

    void cancel() override {
      std::lock_guard<std::mutex> lock(mutex_);
      cancelled_ = true;
      condition_.notify_all();
    }

  private:
    std::mutex mutex_;
    std::condition_variable condition_;
    bool cancelled_ = false;
  };

  auto device = detail::createUsbCIMDevice(
      makeConfig(), std::make_unique<CancelErrorUsbDriver>());
  expectSuccess(device->open());
  device->close();
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
  testReopenPreservesSessionFault();
  testRejectsUnsolicitedAndOverflowResponses();
  testRejectsInvalidTransferConfig();
  testRejectsInvalidEndpointAddresses();
  testRejectsInvalidWordByteOrder();
  testPreservesRawBF16Readback();
  testConsumesCancelledReadError();
  return 0;
}
