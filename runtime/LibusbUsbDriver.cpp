//===- LibusbUsbDriver.cpp - Linux libusb driver ------------*- C++ -*-===//

#include "UsbDriver.h"

#include "llvm/Support/Error.h"

#include <libusb.h>

#include <algorithm>
#include <chrono>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace cim22::runtime::detail {
namespace {
llvm::Error libusbError(const llvm::Twine &operation, int code) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), "%s: %s",
                                 operation.str().c_str(),
                                 libusb_error_name(code));
}

class LibusbUsbDriver final : public UsbDriver {
public:
  ~LibusbUsbDriver() override { close(); }

  llvm::Error open(const UsbCIMDeviceConfig &config) override {
    close();
    config_ = config;
    int result = libusb_init(&context_);
    if (result != LIBUSB_SUCCESS)
      return libusbError("libusb_init", result);

    libusb_device *selected = nullptr;
    libusb_device **devices = nullptr;
    const ssize_t deviceCount = libusb_get_device_list(context_, &devices);
    if (deviceCount < 0) {
      close();
      return libusbError("libusb_get_device_list",
                         static_cast<int>(deviceCount));
    }
    size_t matches = 0;
    for (ssize_t index = 0; index < deviceCount; ++index) {
      libusb_device_descriptor descriptor{};
      if (libusb_get_device_descriptor(devices[index], &descriptor) !=
          LIBUSB_SUCCESS)
        continue;
      if (descriptor.idVendor != config.vendorId ||
          descriptor.idProduct != config.productId ||
          (config.busNumber &&
           libusb_get_bus_number(devices[index]) != *config.busNumber) ||
          (config.deviceAddress &&
           libusb_get_device_address(devices[index]) != *config.deviceAddress))
        continue;
      selected = devices[index];
      ++matches;
    }
    if (matches != 1) {
      libusb_free_device_list(devices, 1);
      close();
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          matches == 0
              ? "CIM USB device %04x:%04x was not found"
              : "CIM USB device %04x:%04x is ambiguous; specify bus/address",
          config.vendorId, config.productId);
    }
    result = libusb_open(selected, &handle_);
    if (result != LIBUSB_SUCCESS || !handle_) {
      libusb_free_device_list(devices, 1);
      close();
      return libusbError("libusb_open", result);
    }

    llvm::Error interfaceError = validateInterface(selected);
    libusb_free_device_list(devices, 1);
    if (interfaceError) {
      close();
      return interfaceError;
    }

    // Linux enforces exclusive interface ownership across handles/processes.
    result = libusb_claim_interface(handle_, config.interfaceNumber);
    if (result != LIBUSB_SUCCESS) {
      close();
      if (result == LIBUSB_ERROR_BUSY)
        return llvm::createStringError(
            llvm::inconvertibleErrorCode(),
            "CIM USB interface %u is already in use by another handle, "
            "process or driver (LIBUSB_ERROR_BUSY)",
            static_cast<unsigned>(config.interfaceNumber));
      return libusbError("libusb_claim_interface", result);
    }
    claimed_ = true;
    uint8_t currentAlternate = 0;
    result = libusb_control_transfer(
        handle_,
        LIBUSB_ENDPOINT_IN | LIBUSB_REQUEST_TYPE_STANDARD |
            LIBUSB_RECIPIENT_INTERFACE,
        LIBUSB_REQUEST_GET_INTERFACE, 0, config.interfaceNumber,
        &currentAlternate, sizeof(currentAlternate), config.transferTimeoutMs);
    if (result != sizeof(currentAlternate)) {
      close();
      if (result < 0)
        return libusbError("libusb GET_INTERFACE", result);
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "libusb GET_INTERFACE returned length %d; "
                                     "expected 1",
                                     result);
    }
    // Reselecting the same alt resets Linux endpoint state and caused IN
    // EPROTO/0B on this FX3 bridge after data; the device-side cause is
    // unknown.
    if (currentAlternate != config.alternateSetting) {
      result = libusb_set_interface_alt_setting(handle_, config.interfaceNumber,
                                                config.alternateSetting);
      if (result != LIBUSB_SUCCESS) {
        close();
        return libusbError("libusb_set_interface_alt_setting", result);
      }
    }
    if (config_.receiveChunkBytes == 0 ||
        config_.receiveChunkBytes >
            static_cast<size_t>(std::numeric_limits<int>::max()) ||
        config_.receivePollTimeoutMs == 0) {
      close();
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "CIM USB receive size or event wait is invalid");
    }
    readBuffer_.resize(config_.receiveChunkBytes);
    readTransfer_ = libusb_alloc_transfer(0);
    if (!readTransfer_) {
      close();
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "libusb_alloc_transfer IN failed");
    }
    libusb_fill_bulk_transfer(
        readTransfer_, handle_, config_.bulkInEndpoint, readBuffer_.data(),
        static_cast<int>(readBuffer_.size()), readCompleted, this, 0);
    int submitResult;
    {
      std::lock_guard<std::mutex> lock(readMutex_);
      stopping_ = false;
      submitResult = submitRead();
    }
    if (submitResult != LIBUSB_SUCCESS) {
      close();
      return libusbError("libusb_submit_transfer IN", submitResult);
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
    const int result = libusb_bulk_transfer(
        handle_, config_.bulkOutEndpoint,
        const_cast<unsigned char *>(bytes.data()), static_cast<int>(request),
        &transferred, config_.transferTimeoutMs);
    if (transferred < 0 || static_cast<size_t>(transferred) > request)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "libusb_bulk_transfer OUT returned invalid length %d; requested=%zu",
          transferred, request);
    if (result != LIBUSB_SUCCESS)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "libusb_bulk_transfer OUT: %s (%d); requested=%zu transferred=%d; "
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
    {
      std::lock_guard<std::mutex> lock(readMutex_);
      stopping_ = true;
      if (readPending_ && !cancelRequested_) {
        const int result = libusb_cancel_transfer(readTransfer_);
        if (result != LIBUSB_SUCCESS && result != LIBUSB_ERROR_NOT_FOUND)
          readError_ = "libusb_cancel_transfer IN: " +
                       std::string(libusb_error_name(result));
        cancelRequested_ = true;
      }
    }
    if (context_)
      libusb_interrupt_event_handler(context_);
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
    // Submitted buffers belong to libusb until the completion callback runs.
    if (readTransfer_)
      libusb_free_transfer(readTransfer_);
    readTransfer_ = nullptr;
    readBuffer_.clear();
    if (handle_) {
      if (claimed_)
        libusb_release_interface(handle_, config_.interfaceNumber);
      libusb_close(handle_);
    }
    handle_ = nullptr;
    claimed_ = false;
    if (context_)
      libusb_exit(context_);
    context_ = nullptr;
    readComplete_ = false;
    readLength_ = 0;
    cancelRequested_ = false;
    readError_.clear();
  }

private:
  llvm::Error validateInterface(libusb_device *device) const {
    libusb_config_descriptor *descriptor = nullptr;
    const int result = libusb_get_active_config_descriptor(device, &descriptor);
    if (result != LIBUSB_SUCCESS)
      return libusbError("libusb_get_active_config_descriptor", result);
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
    libusb_free_config_descriptor(descriptor);
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
    timeval timeout{
        static_cast<time_t>(config_.receivePollTimeoutMs / 1000),
        static_cast<suseconds_t>((config_.receivePollTimeoutMs % 1000) * 1000)};
    return libusb_handle_events_timeout_completed(context_, &timeout, nullptr);
  }

  static void LIBUSB_CALL readCompleted(libusb_transfer *transfer) {
    auto &self = *static_cast<LibusbUsbDriver *>(transfer->user_data);
    std::lock_guard<std::mutex> lock(self.readMutex_);
    self.readPending_ = false;
    self.readComplete_ = true;
    self.readLength_ = transfer->actual_length;
    if (transfer->actual_length < 0 ||
        transfer->actual_length > transfer->length) {
      self.readError_ = "CIM USB IN completion returned invalid length " +
                        std::to_string(transfer->actual_length);
    } else if (transfer->status != LIBUSB_TRANSFER_COMPLETED &&
               !(transfer->status == LIBUSB_TRANSFER_CANCELLED &&
                 self.stopping_)) {
      self.readError_ =
          "CIM USB IN status=" + std::to_string(transfer->status) +
          " requested=" + std::to_string(transfer->length) +
          " transferred=" + std::to_string(transfer->actual_length);
    }
  }

  libusb_context *context_ = nullptr;
  libusb_device_handle *handle_ = nullptr;
  UsbCIMDeviceConfig config_;
  bool claimed_ = false;
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
  return std::make_unique<LibusbUsbDriver>();
}
} // namespace cim22::runtime::detail
