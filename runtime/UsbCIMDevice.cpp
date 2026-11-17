//===- UsbCIMDevice.cpp - CIM22 USB board backend -----------*- C++ -*-===//

#include "CIM22/Runtime/UsbCIMDevice.h"

#include "CIM22/Runtime/CIMPacketCodec.h"
#include "CIM22/Runtime/UsbRecordCodec.h"
#include "UsbDriver.h"
#ifdef CIM22_BOARD_OBSERVER
#include "BoardUsbObserver.h"
#include <filesystem>
#include <fstream>
#endif

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace cim22::runtime {
namespace {
constexpr size_t kResponseRecords = 7;
constexpr size_t kCoreCount = 20;
constexpr size_t kCacheRows = 8;
constexpr size_t kMaxBatchResponses = kCoreCount * kCacheRows;
constexpr size_t kUsbChannelCount = 8;
using ResponsePacket = std::array<uint64_t, kResponseRecords>;
using PendingWeights = std::array<const CIMDeviceWeightWrite *,
                                  execution::CIMWeightResidency::slotCount>;

struct PartialResponse {
  ResponsePacket flits{};
  size_t size = 0;
};

llvm::Error usbError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

#ifdef CIM22_BOARD_OBSERVER
struct UsbCapture {
  explicit UsbCapture(const detail::BoardUsbOptions &options)
      : gate(options.gate), context(options.context) {
    auto directory = std::filesystem::u8path(options.directory);
    planned.open(directory / "planned.bin", std::ios::binary);
    tx.open(directory / "tx.bin", std::ios::binary);
    rx.open(directory / "rx.bin", std::ios::binary);
    weights.open(directory / "weights.csv");
    weights << "slot,begin,end\n";
  }

  bool good() const { return planned && tx && rx && weights; }

  static bool save(std::ofstream &stream, llvm::ArrayRef<uint8_t> bytes) {
    stream.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
    stream.flush();
    return bool(stream);
  }

  detail::BoardUsbGate gate;
  void *context;
  std::ofstream planned, tx, rx, weights;
  uint64_t plannedBytes = 0;
  uint64_t txBytes = 0;
  uint64_t rxBytes = 0;           // Protected by the device receive mutex.
  bool idlePollCompleted = false; // Protected by the device receive mutex.
};
#endif

class UsbCIMDevice final : public CIMDevice {
public:
  UsbCIMDevice(UsbCIMDeviceConfig config,
               std::unique_ptr<detail::UsbDriver> driver)
      : config_(config), driver_(std::move(driver)),
        receiveCodec_(config.wordByteOrder) {}

  ~UsbCIMDevice() override { close(); }

#ifdef CIM22_BOARD_OBSERVER
  void observe(const detail::BoardUsbOptions &options) {
    capture_ = std::make_unique<UsbCapture>(options);
  }
#endif

  llvm::Error open() override {
    std::lock_guard<std::mutex> executionLock(executionMutex_);
    closeUnlocked();
    if (llvm::Error error = validateConfig())
      return error;
    if (!driver_)
      return usbError("CIM USB platform driver is unavailable");
#ifdef CIM22_BOARD_OBSERVER
    if (capture_ && !capture_->good())
      return usbError("Cannot open USB observation files");
#endif
    if (llvm::Error error = driver_->open(config_))
      return error;

    {
      std::lock_guard<std::mutex> lock(receiveMutex_);
      responses_.clear();
      for (PartialResponse &response : partialResponses_)
        response.size = 0;
      responseRecordsRemaining_ = kResponseRecords;
      receiveError_.clear();
      faulted_ = false;
      faultMessage_.clear();
      stopRequested_ = false;
      opened_ = true;
      receiveLoopStarted_ = false;
#ifdef CIM22_BOARD_OBSERVER
      if (capture_)
        capture_->idlePollCompleted = false;
#endif
      receiveCodec_ = UsbRecordCodec(config_.wordByteOrder);
    }
    receiveBuffer_.resize(config_.receiveChunkBytes);
    receiveThread_ = std::thread(&UsbCIMDevice::receiveLoop, this);

    {
      std::unique_lock<std::mutex> lock(receiveMutex_);
      if (!receiveCondition_.wait_for(
              lock, std::chrono::milliseconds(config_.receiveReadyTimeoutMs),
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
#ifdef CIM22_BOARD_OBSERVER
      if (capture_ &&
          (!receiveCondition_.wait_for(
               lock, std::chrono::milliseconds(config_.receiveReadyTimeoutMs),
               [&] {
                 return capture_->idlePollCompleted || capture_->rxBytes ||
                        !receiveError_.empty();
               }) ||
           capture_->rxBytes)) {
        lock.unlock();
        closeUnlocked();
        return usbError("CIM USB same-handle idle IN check failed");
      }
#endif
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
    response.pllVerified.fill(false);
    if (!opened_)
      return usbError("CIM USB device is not open");
    if (faulted_)
      return usbError(llvm::Twine("CIM USB session is faulted: ") +
                      faultMessage_);
    if (!program.useWeightResidency)
      weightResidency_.clear();
    {
      std::lock_guard<std::mutex> lock(receiveMutex_);
      if (hasPendingResponse())
        return fault("CIM USB session has stale response bytes");
      if (!receiveError_.empty())
        return fault(receiveError_);
    }

    auto configurePhase = [&](CIMDevicePhase phase) -> llvm::Error {
      {
        std::lock_guard<std::mutex> lock(receiveMutex_);
        if (!receiveError_.empty())
          return usbError(receiveError_);
      }
      if (config_.configureClock) {
        auto verified = config_.configureClock(phase);
        if (!verified)
          return verified.takeError();
        response.pllVerified[static_cast<size_t>(phase)] = *verified;
      }
      return llvm::Error::success();
    };
    if (llvm::Error error = configurePhase(CIMDevicePhase::Configuration))
      return fault(llvm::toString(std::move(error)));
    bool workPhase = false;
    PendingWeights pendingWeights{};
    for (const CIMDeviceOperation &operation : program.operations) {
      if (operation.phase == CIMDevicePhase::Work && !workPhase) {
        if (llvm::Error error = configurePhase(CIMDevicePhase::Work))
          return fault(llvm::toString(std::move(error)));
#ifdef CIM22_BOARD_OBSERVER
        if (capture_) {
          uint64_t received;
          {
            std::lock_guard<std::mutex> lock(receiveMutex_);
            received = capture_->rxBytes;
            if (hasPendingResponse() || !receiveError_.empty())
              return fault("CIM Work gate found unexpected IN data");
          }
          if (capture_->plannedBytes != capture_->txBytes ||
              capture_->gate(capture_->context, capture_->txBytes, received))
            return fault("CIM Work delivery gate failed");
          std::lock_guard<std::mutex> lock(receiveMutex_);
          if (capture_->rxBytes != received || !receiveError_.empty())
            return fault("CIM Work gate received unexpected IN during capture");
        }
#endif
        workPhase = true;
      } else if (operation.phase != CIMDevicePhase::Work &&
                 (workPhase ||
                  operation.phase != CIMDevicePhase::Configuration)) {
        return fault("CIM device program has an invalid phase order");
      }
      if (operation.weight) {
        if (!program.useWeightResidency)
          return fault("CIM USB deferred Weight operation in raw program");
        if (llvm::Error error = executeWeight(operation, pendingWeights))
          return fault(llvm::toString(std::move(error)));
      } else {
        if (llvm::Error error = executeOperation(operation))
          return fault(llvm::toString(std::move(error)));
      }
    }

    if (!program.readbackBatches.empty())
      if (llvm::Error error = configurePhase(CIMDevicePhase::Readback))
        return fault(llvm::toString(std::move(error)));
    for (const CIMDeviceReadbackBatch &batch : program.readbackBatches) {
      if (llvm::Error error =
              executeReadback(batch, response, program.captureRawReadbacks)) {
        response.readbacks.clear();
        return fault(llvm::toString(std::move(error)));
      }
    }
    {
      std::lock_guard<std::mutex> lock(receiveMutex_);
      if (llvm::Error error = checkReceiveBoundary())
        return fault(llvm::toString(std::move(error)));
    }
    // Confirm staged writes only after the complete response set is verified.
    for (const CIMDeviceWeightWrite *weight : pendingWeights)
      if (weight)
        weightResidency_.store(weight->slot, weight->identity,
                               weight->compensation);
    return llvm::Error::success();
  }

  bool supportsWeightResidency() const override { return true; }

  execution::CIMWeightStats weightStats() override {
    std::lock_guard<std::mutex> lock(executionMutex_);
    return weightResidency_.stats;
  }

  void clearWeightResidency() override {
    std::lock_guard<std::mutex> lock(executionMutex_);
    weightResidency_.clear();
  }

  bool isFaulted() override {
    std::lock_guard<std::mutex> executionLock(executionMutex_);
    std::lock_guard<std::mutex> receiveLock(receiveMutex_);
    return faulted_ || !receiveError_.empty() || hasPendingResponse();
  }

  void close() override {
    std::lock_guard<std::mutex> executionLock(executionMutex_);
    closeUnlocked();
  }

private:
  llvm::Error validateConfig() const {
    if (config_.vendorId == 0 || config_.productId == 0)
      return usbError("CIM USB VID and PID must be non-zero");
#ifndef _WIN32
    if (config_.busNumber.has_value() != config_.deviceAddress.has_value())
      return usbError(
          "CIM USB bus and device address must be specified together");
#endif
    if (config_.bulkInEndpoint == 0 || config_.bulkOutEndpoint == 0 ||
        (config_.bulkInEndpoint & 0x80) == 0 ||
        (config_.bulkOutEndpoint & 0x80) != 0 ||
        (config_.bulkInEndpoint & 0x70) != 0 ||
        (config_.bulkOutEndpoint & 0x70) != 0 ||
        (config_.bulkInEndpoint & 0x0f) == 0 ||
        (config_.bulkOutEndpoint & 0x0f) == 0)
      return usbError("CIM USB bulk endpoint addresses are invalid");
    if (config_.wordByteOrder != UsbWordByteOrder::BigEndian &&
        config_.wordByteOrder != UsbWordByteOrder::LittleEndian)
      return usbError("CIM USB word byte order is invalid");
    if (config_.physicalLink > 3 || config_.logicalChannel > 7)
      return usbError("CIM USB link or channel is outside the record format");
    if (config_.maxTransferBytes == 0 ||
        config_.maxTransferBytes > kUsbRecordBytes * 256 ||
        config_.receiveChunkBytes == 0 || config_.receiveReadyTimeoutMs == 0 ||
        config_.responseTimeoutMs == 0
#ifdef _WIN32
        || config_.windowsAsyncInTimeoutMs == 0 ||
        config_.windowsSyncOutTimeoutMs == 0 ||
        config_.windowsEventWaitTimeoutMs == 0
#else
        || config_.linuxAsyncInTimeoutMs == 0 ||
        config_.linuxSyncOutTimeoutMs == 0 ||
        config_.linuxEventPollTimeoutMs == 0
#endif
    )
      return usbError("CIM USB transfer sizes and timeouts must be non-zero");
    return llvm::Error::success();
  }

  llvm::Error executeOperation(const CIMDeviceOperation &operation) {
    return executeFlits(operation.flits, operation.postDelayMs);
  }

  llvm::Error executeWeight(const CIMDeviceOperation &operation,
                            PendingWeights &pendingWeights) {
    const CIMDeviceWeightWrite &weight = *operation.weight;
    if (!operation.flits.empty() ||
        operation.phase != CIMDevicePhase::Configuration ||
        weight.slot >= execution::CIMWeightResidency::slotCount ||
        !weight.identity)
      return usbError("invalid CIM USB deferred Weight operation");
    // A write staged earlier in this transaction already replaced the
    // physical slot. Prefer it when resolving another write to that slot.
    const CIMDeviceWeightWrite *pending = pendingWeights[weight.slot];
    const bool hit =
        pending ? pending->identity == weight.identity &&
                      pending->compensation == weight.compensation
                : weightResidency_.match(weight.slot, weight.identity,
                                         weight.compensation) != nullptr;
    if (hit) {
      ++weightResidency_.stats.hits;
      return llvm::Error::success();
    }

    llvm::SmallVector<uint32_t, 256> words;
    words.reserve(weight.identity->words.size());
    for (int32_t word : weight.identity->words)
      words.push_back(llvm::bit_cast<uint32_t>(word));
    llvm::SmallVector<uint64_t, 257> flits;
    if (llvm::Error error =
            appendCIMWeightWordsPacket(weight.identity->route, words, flits))
      return error;
#ifdef CIM22_BOARD_OBSERVER
    const uint64_t begin = capture_ ? capture_->txBytes : 0;
#endif
    if (llvm::Error error = executeFlits(flits, operation.postDelayMs))
      return error;
#ifdef CIM22_BOARD_OBSERVER
    if (capture_) {
      capture_->weights << weight.slot << ',' << begin << ','
                        << capture_->txBytes << '\n';
      capture_->weights.flush();
      if (!capture_->weights)
        return usbError("Cannot record completed USB weight write");
    }
#endif
    pendingWeights[weight.slot] = &weight;
    ++weightResidency_.stats.writes;
    return llvm::Error::success();
  }

  llvm::Error executeFlits(llvm::ArrayRef<uint64_t> flits,
                           uint32_t postDelayMs) {
    if (flits.empty())
      return usbError("CIM device operation has no flits");
    if (llvm::Error error = writeFlits(flits))
      return error;
    if (postDelayMs != 0)
      std::this_thread::sleep_for(std::chrono::milliseconds(postDelayMs));
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

#ifdef CIM22_BOARD_OBSERVER
    if (capture_) {
      if (!UsbCapture::save(capture_->planned, bytes))
        return usbError("Cannot preserve planned USB OUT");
      capture_->plannedBytes += bytes.size();
    }
#endif
    size_t offset = 0;
    while (offset < bytes.size()) {
      {
        std::lock_guard<std::mutex> lock(receiveMutex_);
        if (!receiveError_.empty())
          return usbError(receiveError_);
      }
      const size_t request =
          std::min(config_.maxTransferBytes, bytes.size() - offset);
      auto transferred =
          driver_->write(llvm::ArrayRef<uint8_t>(bytes).slice(offset, request));
      if (!transferred)
        return transferred.takeError();
      if (*transferred == 0 || *transferred > request)
        return usbError("CIM USB write returned an invalid transfer length");
#ifdef CIM22_BOARD_OBSERVER
      if (capture_) {
        if (!UsbCapture::save(
                capture_->tx,
                llvm::ArrayRef<uint8_t>(bytes).slice(offset, *transferred)))
          return usbError("Cannot preserve completed USB OUT");
        capture_->txBytes += *transferred;
      }
#endif
      offset += *transferred;
    }
    return llvm::Error::success();
  }

  // Requires receiveMutex_. A response has no Macro identifier, so queued
  // data cannot cross from one selection to the next.
  bool hasPendingResponse() const {
    return !responses_.empty() || receiveCodec_.pendingBytes() != 0 ||
           std::any_of(partialResponses_.begin(), partialResponses_.end(),
                       [](const PartialResponse &response) {
                         return response.size != 0;
                       });
  }

  llvm::Error checkReceiveBoundary() const {
    if (!receiveError_.empty())
      return usbError(receiveError_);
    if (hasPendingResponse())
      return usbError("CIM USB session has unsolicited response bytes");
    return llvm::Error::success();
  }

  // Keep all sends and replies for this Macro under one boundary. A single
  // response deadline prevents one slow packet from extending the budget.
  llvm::Error executeReadback(const CIMDeviceReadbackBatch &batch,
                              CIMDeviceResponse &response,
                              bool captureRawReadbacks) {
    if (batch.targets.empty() || batch.targets.size() > kMaxBatchResponses)
      return usbError("CIM USB readback batch has an invalid target count");
    std::array<std::optional<size_t>, kMaxBatchResponses> outputIndices{};
    for (const CIMDeviceReadbackTarget &target : batch.targets) {
      if (target.coreIndex >= kCoreCount || target.cacheAddress >= kCacheRows)
        return usbError(
            "CIM USB readback target is outside the target profile");
      auto &outputIndex =
          outputIndices[target.coreIndex * kCacheRows + target.cacheAddress];
      if (outputIndex)
        return usbError("CIM USB readback batch has duplicate targets");
      outputIndex = target.outputIndex;
    }
    {
      std::lock_guard<std::mutex> lock(receiveMutex_);
      if (llvm::Error error = checkReceiveBoundary())
        return error;
    }
    if (llvm::Error error = executeOperation(batch.selection))
      return error;
    {
      std::lock_guard<std::mutex> lock(receiveMutex_);
      if (llvm::Error error = checkReceiveBoundary())
        return error;
      responseRecordsRemaining_ = batch.targets.size() * kResponseRecords;
    }
    if (llvm::Error error = executeOperation(batch.request))
      return error;

    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(config_.responseTimeoutMs);
    response.readbacks.reserve(response.readbacks.size() +
                               batch.targets.size());
    for (size_t index = 0; index < batch.targets.size(); ++index) {
      auto records = takeResponse(deadline);
      if (!records)
        return records.takeError();
      const uint64_t header = records->front();
      auto key = decodeCIMOutputCacheResponseHeader(header);
      if (!key)
        return key.takeError();
      auto &outputIndex =
          outputIndices[key->coreIndex * kCacheRows + key->cacheAddress];
      if (!outputIndex)
        return usbError("CIM USB readback batch received an unexpected or "
                        "duplicate Core/Cache response");
      const auto body = llvm::ArrayRef(*records).drop_front();
      if (!captureRawReadbacks)
        if (llvm::Error error = validateCIMOutputCacheResponseBody(body))
          return error;
      CIMDeviceReadback readback;
      readback.outputIndex = *outputIndex;
      readback.headerFlit = header;
      std::copy(body.begin(), body.end(), readback.bodyFlits.begin());
      outputIndex.reset();
      response.readbacks.push_back(readback);
    }
    std::lock_guard<std::mutex> lock(receiveMutex_);
    return checkReceiveBoundary();
  }

  llvm::Expected<ResponsePacket>
  takeResponse(std::chrono::steady_clock::time_point deadline) {
    std::unique_lock<std::mutex> lock(receiveMutex_);
    if (!receiveCondition_.wait_until(lock, deadline, [&] {
          return !responses_.empty() || responseRecordsRemaining_ == 0 ||
                 !receiveError_.empty() || stopRequested_;
        }))
      return usbError("CIM USB response timed out");
    if (!receiveError_.empty())
      return usbError(receiveError_);
    if (responses_.empty() && responseRecordsRemaining_ == 0)
      return usbError("CIM USB response has incomplete channel packets");
    if (responses_.empty())
      return usbError(
          "CIM USB receive loop stopped before a complete response");

    const ResponsePacket result = responses_.front();
    responses_.pop_front();
    return result;
  }

  // After a transport fault the chip's last completed operation is unknown.
  // Forget cached state and block further execute calls until open.
  llvm::Error fault(std::string message) {
    weightResidency_.clear();
    faulted_ = true;
    faultMessage_ = std::move(message);
    return usbError(faultMessage_);
  }

  void receiveLoop() {
    auto &buffer = receiveBuffer_;
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
        // Drivers distinguish expected cancellation from genuine failures.
        receiveError_ = llvm::toString(count.takeError());
        receiveCondition_.notify_all();
        break;
      }
      if (*count == 0) {
#ifdef CIM22_BOARD_OBSERVER
        if (capture_) {
          std::lock_guard<std::mutex> lock(receiveMutex_);
          if (!capture_->idlePollCompleted) {
            capture_->idlePollCompleted = true;
            receiveCondition_.notify_all();
          }
        }
#endif
        continue;
      }
      if (!appendReceived(buffer, *count))
        break;
    }
  }

  bool appendReceived(llvm::ArrayRef<uint8_t> buffer, size_t count) {
    std::lock_guard<std::mutex> lock(receiveMutex_);
    receiveCondition_.notify_all();
    auto fail = [&](std::string message) {
      if (receiveError_.empty())
        receiveError_ = std::move(message);
      return false;
    };
    if (count > buffer.size())
      return fail("CIM USB read returned an invalid transfer length");
#ifdef CIM22_BOARD_OBSERVER
    if (capture_) {
      if (!UsbCapture::save(capture_->rx, buffer.take_front(count)))
        return fail("Cannot preserve completed USB IN");
      capture_->rxBytes += count;
    }
#endif
    llvm::SmallVector<UsbRecord, 16> decoded;
    if (auto error = receiveCodec_.append(buffer.take_front(count), decoded)) {
      return fail(llvm::toString(std::move(error)));
    }
    if (std::any_of(
            decoded.begin(), decoded.end(),
            [](const UsbRecord &record) { return record.physicalLink != 0; }))
      return fail("CIM USB response has non-zero reserved link bits");
    if (decoded.size() > responseRecordsRemaining_) {
      return fail("CIM USB response queue overflow");
    }
    responseRecordsRemaining_ -= decoded.size();
    // Packet flits stay ordered within a logical channel. Other channels may
    // interleave, and neither packet completion nor Core order is prescribed.
    for (const UsbRecord &record : decoded) {
      PartialResponse &packet = partialResponses_[record.logicalChannel];
      packet.flits[packet.size++] = record.flit;
      if (packet.size == kResponseRecords) {
        responses_.push_back(packet.flits);
        packet.size = 0;
      }
    }
    return true;
  }

  void closeUnlocked() {
    weightResidency_.clear();
    {
      std::lock_guard<std::mutex> lock(receiveMutex_);
      stopRequested_ = true;
      receiveCondition_.notify_all();
    }
    if (driver_)
      driver_->cancel();
    if (receiveThread_.joinable())
      receiveThread_.join();
    if (driver_ && opened_) {
      // Retire the outstanding request before releasing its endpoint. Keep
      // tail records and errors observable by the final isFaulted() check.
      std::vector<uint8_t> buffer(config_.receiveChunkBytes);
      while (true) {
        auto count = driver_->drain(buffer);
        if (!count) {
          std::lock_guard<std::mutex> lock(receiveMutex_);
          auto message = llvm::toString(count.takeError());
          if (receiveError_.empty())
            receiveError_ = std::move(message);
          break;
        }
        if (*count == 0 || !appendReceived(buffer, *count))
          break;
      }
    }
    if (driver_)
      driver_->close();
    opened_ = false;
  }

  UsbCIMDeviceConfig config_;
  execution::CIMWeightResidency weightResidency_;
  std::unique_ptr<detail::UsbDriver> driver_;
  std::mutex executionMutex_;
  std::mutex receiveMutex_;
  std::condition_variable receiveCondition_;
  std::thread receiveThread_;
  std::vector<uint8_t> receiveBuffer_;
  UsbRecordCodec receiveCodec_;
  std::deque<ResponsePacket> responses_;
  std::array<PartialResponse, kUsbChannelCount> partialResponses_{};
  // Budget for the entire response set, including packets already consumed.
  size_t responseRecordsRemaining_ = kResponseRecords;
  std::string receiveError_;
  std::string faultMessage_;
  bool stopRequested_ = true;
  bool receiveLoopStarted_ = false;
  bool opened_ = false;
  bool faulted_ = false;
#ifdef CIM22_BOARD_OBSERVER
  std::unique_ptr<UsbCapture> capture_;
#endif
};
} // namespace

std::unique_ptr<CIMDevice>
createUsbCIMDevice(const UsbCIMDeviceConfig &config) {
  return std::make_unique<UsbCIMDevice>(config,
                                        detail::createPlatformUsbDriver());
}

namespace detail {
#ifdef CIM22_BOARD_OBSERVER
std::unique_ptr<CIMDevice>
createObservedUsbDevice(const UsbCIMDeviceConfig &config,
                        const BoardUsbOptions &options) {
  auto device =
      std::make_unique<UsbCIMDevice>(config, createPlatformUsbDriver());
  device->observe(options);
  return device;
}
#endif
std::unique_ptr<CIMDevice>
createUsbCIMDevice(const UsbCIMDeviceConfig &config,
                   std::unique_ptr<UsbDriver> driver) {
  return std::make_unique<UsbCIMDevice>(config, std::move(driver));
}
} // namespace detail
} // namespace cim22::runtime
