//===- UsbCIMDevice.cpp - CIM22 USB board backend -----------*- C++ -*-===//

#include "CIM22/Runtime/UsbCIMDevice.h"

#include "CIM22/Runtime/Detail/UsbDriver.h"
#include "CIM22/Runtime/UsbRecordCodec.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace cim22::runtime {
namespace {
constexpr size_t kMaxResponseRecords = 7;

llvm::Error usbError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

class UsbCIMDevice final : public CIMDevice {
public:
  UsbCIMDevice(UsbCIMDeviceConfig config,
               std::unique_ptr<detail::UsbDriver> driver)
      : config_(config), driver_(std::move(driver)),
        receiveCodec_(config.wordByteOrder) {}

  ~UsbCIMDevice() override { close(); }

  llvm::Error open() override {
    std::lock_guard<std::mutex> executionLock(executionMutex_);
    closeUnlocked();
    if (faulted_ || !receiveError_.empty() || !records_.empty() ||
        receiveCodec_.pendingBytes() != 0) {
      faulted_ = true;
      return usbError("CIM USB session remains faulted; close the Session, "
                      "reset FPGA/chip, then create a new Session");
    }
    if (llvm::Error error = validateConfig())
      return error;
    if (!driver_)
      return usbError("CIM USB platform driver is unavailable");
    if (llvm::Error error = driver_->open(config_))
      return error;

    {
      std::lock_guard<std::mutex> lock(receiveMutex_);
      records_.clear();
      receiveError_.clear();
      stopRequested_ = false;
      opened_ = true;
      receiveLoopStarted_ = false;
      receiveCodec_ = UsbRecordCodec(config_.wordByteOrder);
    }
    receiveThread_ = std::thread(&UsbCIMDevice::receiveLoop, this);

    {
      std::unique_lock<std::mutex> lock(receiveMutex_);
      if (!receiveCondition_.wait_for(
              lock, std::chrono::milliseconds(config_.transferTimeoutMs),
              [&] { return receiveLoopStarted_ || !receiveError_.empty(); })) {
        lock.unlock();
        closeUnlocked();
        return usbError("CIM USB receive loop did not start");
      }
      if (!receiveError_.empty()) {
        std::string message = receiveError_;
        lock.unlock();
        closeUnlocked();
        return usbError(message);
      }
    }
    if (config_.startupDelayMs != 0)
      std::this_thread::sleep_for(
          std::chrono::milliseconds(config_.startupDelayMs));
    {
      std::unique_lock<std::mutex> lock(receiveMutex_);
      if (!receiveError_.empty()) {
        std::string message = receiveError_;
        lock.unlock();
        closeUnlocked();
        return usbError(message);
      }
    }
    return llvm::Error::success();
  }

  llvm::Error execute(const CIMDeviceProgram &program,
                      CIMDeviceResponse &response) override {
    std::lock_guard<std::mutex> executionLock(executionMutex_);
    response.readbacks.clear();
    if (!opened_)
      return usbError("CIM USB device is not open");
    if (faulted_)
      return usbError(llvm::Twine("CIM USB session is faulted: ") +
                      faultMessage_);
    for (const CIMDeviceReadbackCommand &readback : program.readbacks)
      if (readback.dataType == execution::CIMDataType::BF16)
        return usbError(
            "CIM USB BF16 readback is disabled until the board response ABI "
            "is verified");

    {
      std::lock_guard<std::mutex> lock(receiveMutex_);
      if (!records_.empty() || receiveCodec_.pendingBytes() != 0)
        return fault("CIM USB session has stale response bytes");
      if (!receiveError_.empty())
        return fault(receiveError_);
    }

    for (const CIMDeviceOperation &operation : program.operations)
      if (llvm::Error error = executeOperation(operation))
        return fault(llvm::toString(std::move(error)));

    response.readbacks.reserve(program.readbacks.size());
    for (const CIMDeviceReadbackCommand &command : program.readbacks) {
      if (llvm::Error error = executeOperation(command.request))
        return fault(llvm::toString(std::move(error)));
      auto records = takeRecords(kMaxResponseRecords);
      if (!records)
        return fault(llvm::toString(records.takeError()));

      CIMDeviceReadback readback;
      readback.outputIndex = command.outputIndex;
      readback.headerFlit = (*records)[0].flit;
      for (size_t index = 0; index < readback.bodyFlits.size(); ++index)
        readback.bodyFlits[index] = (*records)[index + 1].flit;
      response.readbacks.push_back(readback);
    }
    return llvm::Error::success();
  }

  bool isFaulted() override {
    std::lock_guard<std::mutex> executionLock(executionMutex_);
    std::lock_guard<std::mutex> receiveLock(receiveMutex_);
    return faulted_ || !receiveError_.empty() || !records_.empty() ||
           receiveCodec_.pendingBytes() != 0;
  }

  void close() override {
    std::lock_guard<std::mutex> executionLock(executionMutex_);
    closeUnlocked();
  }

private:
  llvm::Error validateConfig() const {
    if (config_.vendorId == 0 || config_.productId == 0)
      return usbError("CIM USB VID and PID must be non-zero");
    if (config_.bulkInEndpoint == 0 || config_.bulkOutEndpoint == 0 ||
        (config_.bulkInEndpoint & 0x80) == 0 ||
        (config_.bulkOutEndpoint & 0x80) != 0 ||
        (config_.bulkInEndpoint & 0x0f) == 0 ||
        (config_.bulkOutEndpoint & 0x0f) == 0)
      return usbError("CIM USB bulk endpoint directions are invalid");
    if (config_.physicalLink > 3 || config_.logicalChannel > 7)
      return usbError("CIM USB link or channel is outside the record format");
    if (config_.maxTransferBytes == 0 ||
        config_.maxTransferBytes > kUsbRecordBytes * 256 ||
        config_.receiveChunkBytes == 0 || config_.transferTimeoutMs == 0 ||
        config_.receivePollTimeoutMs == 0 || config_.responseTimeoutMs == 0)
      return usbError("CIM USB transfer sizes and timeouts must be non-zero");
    return llvm::Error::success();
  }

  llvm::Error executeOperation(const CIMDeviceOperation &operation) {
    if (operation.flits.empty())
      return usbError("CIM device operation has no flits");
    if (llvm::Error error = writeFlits(operation.flits))
      return error;
    if (operation.postDelayMs != 0)
      std::this_thread::sleep_for(
          std::chrono::milliseconds(operation.postDelayMs));
    return llvm::Error::success();
  }

  llvm::Error writeFlits(llvm::ArrayRef<uint64_t> flits) {
    std::vector<uint8_t> bytes;
    bytes.reserve(flits.size() * kUsbRecordBytes);
    for (uint64_t flit : flits) {
      const UsbRecord record{config_.physicalLink, config_.logicalChannel,
                             flit};
      if (llvm::Error error = UsbRecordCodec::validate(record))
        return error;
      const auto encoded =
          UsbRecordCodec::encode(record, config_.wordByteOrder);
      bytes.insert(bytes.end(), encoded.begin(), encoded.end());
    }

    size_t offset = 0;
    while (offset < bytes.size()) {
      const size_t request =
          std::min(config_.maxTransferBytes, bytes.size() - offset);
      auto transferred =
          driver_->write(llvm::ArrayRef<uint8_t>(bytes).slice(offset, request));
      if (!transferred)
        return transferred.takeError();
      if (*transferred == 0 || *transferred > request)
        return usbError("CIM USB write returned an invalid transfer length");
      offset += *transferred;
    }
    return llvm::Error::success();
  }

  llvm::Expected<std::vector<UsbRecord>> takeRecords(size_t count) {
    std::unique_lock<std::mutex> lock(receiveMutex_);
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(config_.responseTimeoutMs);
    if (!receiveCondition_.wait_until(lock, deadline, [&] {
          return records_.size() >= count || !receiveError_.empty() ||
                 stopRequested_;
        }))
      return usbError("CIM USB response timed out");
    if (!receiveError_.empty())
      return usbError(receiveError_);
    if (records_.size() < count)
      return usbError(
          "CIM USB receive loop stopped before a complete response");

    std::vector<UsbRecord> result;
    result.reserve(count);
    for (size_t index = 0; index < count; ++index) {
      result.push_back(records_.front());
      records_.pop_front();
    }
    return result;
  }

  llvm::Error fault(std::string message) {
    faulted_ = true;
    faultMessage_ = std::move(message);
    return usbError(faultMessage_);
  }

  void receiveLoop() {
    std::vector<uint8_t> buffer(config_.receiveChunkBytes);
    {
      std::lock_guard<std::mutex> lock(receiveMutex_);
      receiveLoopStarted_ = true;
      receiveCondition_.notify_all();
    }
    while (true) {
      {
        std::lock_guard<std::mutex> lock(receiveMutex_);
        if (stopRequested_)
          break;
      }
      auto count = driver_->read(buffer);
      if (!count) {
        std::lock_guard<std::mutex> lock(receiveMutex_);
        if (!stopRequested_)
          receiveError_ = llvm::toString(count.takeError());
        receiveCondition_.notify_all();
        break;
      }
      if (*count == 0)
        continue;
      if (*count > buffer.size()) {
        std::lock_guard<std::mutex> lock(receiveMutex_);
        receiveError_ = "CIM USB read returned an invalid transfer length";
        receiveCondition_.notify_all();
        break;
      }

      llvm::SmallVector<UsbRecord, 16> decoded;
      std::lock_guard<std::mutex> lock(receiveMutex_);
      if (llvm::Error error = receiveCodec_.append(
              llvm::ArrayRef<uint8_t>(buffer).take_front(*count), decoded)) {
        receiveError_ = llvm::toString(std::move(error));
        receiveCondition_.notify_all();
        break;
      }
      if (decoded.size() > kMaxResponseRecords - records_.size()) {
        receiveError_ = "CIM USB response queue overflow";
        receiveCondition_.notify_all();
        break;
      }
      records_.insert(records_.end(), decoded.begin(), decoded.end());
      receiveCondition_.notify_all();
    }
  }

  void closeUnlocked() {
    {
      std::lock_guard<std::mutex> lock(receiveMutex_);
      stopRequested_ = true;
      receiveCondition_.notify_all();
    }
    if (driver_)
      driver_->cancel();
    if (receiveThread_.joinable())
      receiveThread_.join();
    if (driver_)
      driver_->close();
    opened_ = false;
  }

  UsbCIMDeviceConfig config_;
  std::unique_ptr<detail::UsbDriver> driver_;
  std::mutex executionMutex_;
  std::mutex receiveMutex_;
  std::condition_variable receiveCondition_;
  std::thread receiveThread_;
  UsbRecordCodec receiveCodec_;
  std::deque<UsbRecord> records_;
  std::string receiveError_;
  std::string faultMessage_;
  bool stopRequested_ = true;
  bool receiveLoopStarted_ = false;
  bool opened_ = false;
  bool faulted_ = false;
};
} // namespace

std::unique_ptr<CIMDevice>
createUsbCIMDevice(const UsbCIMDeviceConfig &config) {
  return std::make_unique<UsbCIMDevice>(config,
                                        detail::createPlatformUsbDriver());
}

namespace detail {
std::unique_ptr<CIMDevice>
createUsbCIMDevice(const UsbCIMDeviceConfig &config,
                   std::unique_ptr<UsbDriver> driver) {
  return std::make_unique<UsbCIMDevice>(config, std::move(driver));
}
} // namespace detail
} // namespace cim22::runtime
