//===- LibcyusbUsbDriver.cpp - Linux CyUSB driver -------------*- C++ -*-===//

#include "UsbDriver.h"

#include "llvm/Support/Error.h"

#include <cyusb.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace cim22::runtime::detail {
namespace {
std::atomic_flag cyusbSessionActive = ATOMIC_FLAG_INIT;

struct QuarantinedTransfer {
  libusb_transfer *transfer = nullptr;
  std::vector<uint8_t> buffer;
  cyusb_handle *handle = nullptr;
  int interfaceNumber = 0;
  bool claimed = false;
  bool detached = false;
};

void LIBUSB_CALL completeQuarantinedTransfer(libusb_transfer *transfer) {
  auto *quarantine = static_cast<QuarantinedTransfer *>(transfer->user_data);
  if (quarantine) {
    if (quarantine->claimed)
      cyusb_release_interface(quarantine->handle, quarantine->interfaceNumber);
    if (quarantine->detached)
      cyusb_attach_kernel_driver(quarantine->handle,
                                 quarantine->interfaceNumber);
    cyusb_close();
    delete quarantine;
    cyusbSessionActive.clear(std::memory_order_release);
  }
  libusb_free_transfer(transfer);
}

llvm::Error cyusbError(const llvm::Twine &operation, int code) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), "%s: %s (%d)",
                                 operation.str().c_str(),
                                 libusb_error_name(code), code);
}

class LibcyusbUsbDriver final : public UsbDriver {
public:
  ~LibcyusbUsbDriver() override { close(); }

  llvm::Error open(const UsbCIMDeviceConfig &config) override {
    close();
    if (config.maxTransferBytes == 0 ||
        config.maxTransferBytes >
            static_cast<size_t>(std::numeric_limits<int>::max()) ||
        config.receiveReadyTimeoutMs == 0 ||
        config.linuxAsyncInTimeoutMs == 0 ||
        config.linuxSyncOutTimeoutMs == 0 ||
        config.linuxEventPollTimeoutMs == 0 ||
        config.linuxAsyncInTimeoutMs >
            static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
        config.receiveChunkBytes == 0 ||
        config.receiveChunkBytes >
            static_cast<size_t>(std::numeric_limits<int>::max()) ||
        config.linuxSyncOutTimeoutMs >
            static_cast<uint32_t>(std::numeric_limits<int>::max()))
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "CIM CyUSB transfer size or timeout is invalid");

    if (cyusbSessionActive.test_and_set(std::memory_order_acquire))
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "another CIM CyUSB session is already active in this process");
    sessionActive_ = true;

    config_ = config;
    const int count = cyusb_open(config.vendorId, config.productId);
    if (count <= 0)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "cyusb_open could not find device %04x:%04x (result=%d)",
          config.vendorId, config.productId, count);
    handle_ = cyusb_gethandle(0);
    if (!handle_) {
      cyusb_close();
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "cyusb_gethandle returned null");
    }
    if ((config.busNumber &&
         cyusb_get_busnumber(handle_) != *config.busNumber) ||
        (config.deviceAddress &&
         cyusb_get_devaddr(handle_) != *config.deviceAddress)) {
      const int bus = cyusb_get_busnumber(handle_);
      const int address = cyusb_get_devaddr(handle_);
      close();
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "cyusb_open selected bus=%d address=%d, requested bus/address do not "
          "match; cyusb_open(VID,PID) cannot select among duplicates",
          bus, address);
    }
    if (llvm::Error error = validateInterface()) {
      close();
      return error;
    }
    int result = cyusb_kernel_driver_active(handle_, config.interfaceNumber);
    if (result < 0) {
      close();
      return cyusbError("cyusb_kernel_driver_active", result);
    }
    if (result == 1) {
      result = cyusb_detach_kernel_driver(handle_, config.interfaceNumber);
      if (result != 0) {
        close();
        return cyusbError("cyusb_detach_kernel_driver", result);
      }
      detached_ = true;
    }
    result = cyusb_claim_interface(handle_, config.interfaceNumber);
    if (result != 0) {
      close();
      return cyusbError("cyusb_claim_interface", result);
    }
    claimed_ = true;
    uint8_t currentAlternate = 0;
    result = libusb_control_transfer(
        handle_,
        LIBUSB_ENDPOINT_IN | LIBUSB_REQUEST_TYPE_STANDARD |
            LIBUSB_RECIPIENT_INTERFACE,
        LIBUSB_REQUEST_GET_INTERFACE, 0, config.interfaceNumber,
        &currentAlternate, sizeof(currentAlternate),
        config.linuxSyncOutTimeoutMs);
    if (result != sizeof(currentAlternate)) {
      close();
      if (result < 0)
        return cyusbError("libusb GET_INTERFACE", result);
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "libusb GET_INTERFACE returned length %d; expected 1", result);
    }
    if (currentAlternate != config.alternateSetting) {
      result = cyusb_set_interface_alt_setting(handle_, config.interfaceNumber,
                                               config.alternateSetting);
      if (result != 0) {
        close();
        return cyusbError("cyusb_set_interface_alt_setting", result);
      }
    }
    readBuffer_.resize(config.receiveChunkBytes);
    readTransfer_ = libusb_alloc_transfer(0);
    if (!readTransfer_) {
      close();
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "libusb_alloc_transfer IN failed");
    }
    libusb_fill_bulk_transfer(
        readTransfer_, handle_, config.bulkInEndpoint, readBuffer_.data(),
        static_cast<int>(readBuffer_.size()), readCompleted, this,
        config.linuxAsyncInTimeoutMs);
    int submitResult = LIBUSB_SUCCESS;
    {
      std::lock_guard<std::mutex> lock(readMutex_);
      stopping_ = false;
      cancelRequested_ = false;
      readError_.clear();
      readComplete_ = false;
      submitResult = submitRead();
    }
    if (submitResult != LIBUSB_SUCCESS) {
      const std::string message = std::string("libusb_submit_transfer IN: ") +
                                  libusb_error_name(submitResult);
      close();
      return llvm::createStringError(llvm::inconvertibleErrorCode(), "%s",
                                     message.c_str());
    }
    return llvm::Error::success();
  }

  llvm::Expected<size_t> write(llvm::ArrayRef<uint8_t> bytes) override {
    if (bytes.empty())
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "CIM USB write cannot be empty");
    {
      std::lock_guard<std::mutex> lock(readMutex_);
      if (!readError_.empty())
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       readError_);
      if (!handle_ || stopping_)
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "CIM USB device is not open");
    }
    const size_t request = std::min(
        bytes.size(), static_cast<size_t>(std::numeric_limits<int>::max()));
    int transferred = 0;
    // A synchronous OUT can dispatch the IN callback on this thread.
    const int result = cyusb_bulk_transfer(
        handle_, config_.bulkOutEndpoint,
        const_cast<unsigned char *>(bytes.data()), static_cast<int>(request),
        &transferred, config_.linuxSyncOutTimeoutMs);
    if (transferred < 0 || static_cast<size_t>(transferred) > request)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "cyusb_bulk_transfer OUT returned invalid length %d; requested=%zu",
          transferred, request);
    if (result != LIBUSB_SUCCESS)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "cyusb_bulk_transfer OUT: %s (%d); requested=%zu transferred=%d; "
          "failed prefix is not retried",
          libusb_error_name(result), result, request, transferred);
    {
      std::lock_guard<std::mutex> lock(readMutex_);
      if (!readError_.empty())
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       readError_);
    }
    return static_cast<size_t>(transferred);
  }

  llvm::Expected<size_t> read(llvm::MutableArrayRef<uint8_t> bytes) override {
    if (bytes.empty())
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "CIM USB read buffer cannot be empty");
    while (true) {
      {
        std::lock_guard<std::mutex> lock(readMutex_);
        if (!handle_)
          return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                         "CIM USB device is not open");
        if (!readError_.empty())
          return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                         readError_);
        if (readComplete_) {
          readComplete_ = false;
          const size_t count = static_cast<size_t>(readLength_);
          if (count > bytes.size()) {
            readError_ = "CIM USB IN completion exceeds the caller buffer";
            return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                           readError_);
          }
          std::copy_n(readBuffer_.begin(), count, bytes.begin());
          if (!stopping_) {
            const int result = submitRead();
            if (result != LIBUSB_SUCCESS) {
              readError_ = "libusb_submit_transfer IN: " +
                           std::string(libusb_error_name(result));
              return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                             readError_);
            }
          }
          return count;
        }
        if (stopping_ && !readPending_)
          return size_t{0};
      }
      const int result = pumpEvents();
      if (result != LIBUSB_SUCCESS && result != LIBUSB_ERROR_INTERRUPTED) {
        std::lock_guard<std::mutex> lock(readMutex_);
        readError_ = "libusb_handle_events IN: " +
                     std::string(libusb_error_name(result));
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       readError_);
      }
    }
  }

  void cancel() override {
    bool interrupt = false;
    {
      std::lock_guard<std::mutex> lock(readMutex_);
      stopping_ = true;
      if (readPending_ && !cancelRequested_) {
        const int result = libusb_cancel_transfer(readTransfer_);
        if (result != LIBUSB_SUCCESS && result != LIBUSB_ERROR_NOT_FOUND) {
          readError_ = "libusb_cancel_transfer IN: " +
                       std::string(libusb_error_name(result));
          // A hard cancel failure has no callback to drive cleanup. Treat the
          // transfer as terminal so close() cannot wait forever. A hard
          // cancellation error does not guarantee that the transfer stopped;
          // quarantine its storage until libusb eventually invokes a callback.
          readPending_ = false;
          quarantineTransferLocked();
        }
        cancelRequested_ = true;
        interrupt = true;
      }
    }
    if (interrupt)
      libusb_interrupt_event_handler(nullptr);
  }

  llvm::Expected<size_t> drain(llvm::MutableArrayRef<uint8_t> bytes) override {
    return read(bytes);
  }

  void close() override {
    cancel();
    while (true) {
      {
        std::lock_guard<std::mutex> lock(readMutex_);
        if (!readPending_)
          break;
      }
      const int result = pumpEvents();
      if (result != LIBUSB_SUCCESS && result != LIBUSB_ERROR_INTERRUPTED)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (readTransfer_)
      libusb_free_transfer(readTransfer_);
    readTransfer_ = nullptr;
    readBuffer_.clear();
    if (handle_) {
      if (claimed_)
        cyusb_release_interface(handle_, config_.interfaceNumber);
      if (detached_)
        cyusb_attach_kernel_driver(handle_, config_.interfaceNumber);
      cyusb_close();
    }
    handle_ = nullptr;
    claimed_ = false;
    detached_ = false;
    {
      std::lock_guard<std::mutex> lock(readMutex_);
      readComplete_ = false;
      readLength_ = 0;
      readPending_ = false;
      cancelRequested_ = false;
      stopping_ = true;
      readError_.clear();
    }
    if (sessionActive_) {
      sessionActive_ = false;
      cyusbSessionActive.clear(std::memory_order_release);
    }
  }

private:
  llvm::Error validateInterface() const {
    libusb_config_descriptor *descriptor = nullptr;
    const int result = cyusb_get_active_config_descriptor(handle_, &descriptor);
    if (result != LIBUSB_SUCCESS)
      return cyusbError("libusb_get_active_config_descriptor", result);
    const libusb_interface_descriptor *interface = nullptr;
    for (int index = 0; index < descriptor->bNumInterfaces; ++index) {
      const libusb_interface &candidate = descriptor->interface[index];
      for (int alt = 0; alt < candidate.num_altsetting; ++alt) {
        const libusb_interface_descriptor &setting = candidate.altsetting[alt];
        if (setting.bInterfaceNumber == config_.interfaceNumber &&
            setting.bAlternateSetting == config_.alternateSetting) {
          interface = &setting;
          break;
        }
      }
    }
    const bool interfaceFound = interface != nullptr;
    bool foundIn = false;
    bool foundOut = false;
    if (interface) {
      for (uint8_t index = 0; index < interface->bNumEndpoints; ++index) {
        const libusb_endpoint_descriptor &endpoint = interface->endpoint[index];
        if ((endpoint.bmAttributes & LIBUSB_TRANSFER_TYPE_MASK) !=
            LIBUSB_TRANSFER_TYPE_BULK)
          continue;
        foundIn |= endpoint.bEndpointAddress == config_.bulkInEndpoint;
        foundOut |= endpoint.bEndpointAddress == config_.bulkOutEndpoint;
      }
    }
    cyusb_free_config_descriptor(descriptor);
    if (!interfaceFound || !foundIn || !foundOut)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "CIM USB interface/alternate setting "
                                     "lacks configured bulk endpoints");
    return llvm::Error::success();
  }

  // Called with readMutex_ held; submit does not invoke the callback inline.
  int submitRead() {
    const int result = libusb_submit_transfer(readTransfer_);
    readPending_ = result == LIBUSB_SUCCESS;
    return result;
  }

  int pumpEvents() {
    timeval timeout{static_cast<time_t>(config_.linuxEventPollTimeoutMs / 1000),
                    static_cast<suseconds_t>(
                        (config_.linuxEventPollTimeoutMs % 1000) * 1000)};
    return libusb_handle_events_timeout_completed(nullptr, &timeout, nullptr);
  }

  void quarantineTransferLocked() {
    auto *quarantine = new QuarantinedTransfer;
    quarantine->transfer = readTransfer_;
    quarantine->buffer = std::move(readBuffer_);
    quarantine->handle = handle_;
    quarantine->interfaceNumber = config_.interfaceNumber;
    quarantine->claimed = claimed_;
    quarantine->detached = detached_;
    quarantine->transfer->buffer = quarantine->buffer.data();
    quarantine->transfer->callback = completeQuarantinedTransfer;
    quarantine->transfer->user_data = quarantine;
    readTransfer_ = nullptr;
    readBuffer_.clear();
    handle_ = nullptr;
    claimed_ = false;
    detached_ = false;
    // The callback owns the process-global session reservation from this point.
    sessionActive_ = false;
  }

  static void LIBUSB_CALL readCompleted(libusb_transfer *transfer) {
    auto &self = *static_cast<LibcyusbUsbDriver *>(transfer->user_data);
    std::lock_guard<std::mutex> lock(self.readMutex_);
    self.readPending_ = false;
    self.readLength_ = transfer->actual_length;
    if (transfer->actual_length < 0 ||
        transfer->actual_length > transfer->length) {
      self.readError_ = "CIM USB IN completion returned invalid length " +
                        std::to_string(transfer->actual_length);
    } else if (transfer->status != LIBUSB_TRANSFER_COMPLETED &&
               transfer->status != LIBUSB_TRANSFER_TIMED_OUT &&
               !(transfer->status == LIBUSB_TRANSFER_CANCELLED &&
                 self.stopping_)) {
      self.readError_ =
          "CIM USB IN status=" + std::to_string(transfer->status) +
          " requested=" + std::to_string(transfer->length) +
          " transferred=" + std::to_string(transfer->actual_length);
    } else {
      // Deliver idle polls and any partial data before reusing the buffer.
      self.readComplete_ = true;
    }
  }

  libusb_device_handle *handle_ = nullptr;
  UsbCIMDeviceConfig config_;
  bool claimed_ = false;
  bool detached_ = false;
  bool sessionActive_ = false;
  std::mutex readMutex_;
  libusb_transfer *readTransfer_ = nullptr;
  std::vector<uint8_t> readBuffer_;
  std::string readError_;
  int readLength_ = 0;
  bool readPending_ = false;
  bool readComplete_ = false;
  bool stopping_ = true;
  bool cancelRequested_ = false;
};
} // namespace

std::unique_ptr<UsbDriver> createPlatformUsbDriver() {
  return std::make_unique<LibcyusbUsbDriver>();
}
} // namespace cim22::runtime::detail
