//===- LibcyusbUsbDriver.cpp - Linux CyUSB driver -------------*- C++ -*-===//

#include "UsbDriver.h"

#include "llvm/Support/Error.h"

#include <cyusb.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

namespace cim22::runtime::detail {
namespace {

std::mutex cyusbSessionMutex;

llvm::Error cyusbError(const llvm::Twine &operation, int code) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), "%s: %s (%d)",
                                 operation.str().c_str(),
                                 libusb_error_name(code), code);
}

// shortcut: libcyusb owns process-global handles; this runtime permits one
// active CyUSB session per process until a multi-client adapter is required.
class LibcyusbUsbDriver final : public UsbDriver {
public:
  ~LibcyusbUsbDriver() override { close(); }

  llvm::Error open(const UsbCIMDeviceConfig &config) override {
    close();
    if (config.maxTransferBytes == 0 ||
        config.maxTransferBytes >
            static_cast<size_t>(std::numeric_limits<int>::max()) ||
        config.transferTimeoutMs == 0 ||
        config.transferTimeoutMs >
            static_cast<uint32_t>(std::numeric_limits<int>::max()))
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "CIM CyUSB transfer size or timeout is invalid");

    std::unique_lock<std::mutex> sessionLock(cyusbSessionMutex,
                                             std::try_to_lock);
    if (!sessionLock.owns_lock())
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "another CIM CyUSB session is already active in this process");

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
        &currentAlternate, sizeof(currentAlternate), config.transferTimeoutMs);
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

    {
      std::lock_guard<std::mutex> lock(stateMutex_);
      stopping_ = false;
      opened_ = true;
    }
    sessionLock_ = std::move(sessionLock);
    return llvm::Error::success();
  }

  llvm::Expected<size_t> write(llvm::ArrayRef<uint8_t> bytes) override {
    if (bytes.empty())
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "CIM USB write cannot be empty");
    if (llvm::Error error = checkOpen())
      return std::move(error);

    std::lock_guard<std::mutex> transferLock(transferMutex_);
    if (llvm::Error error = checkOpen())
      return std::move(error);
    const size_t request = std::min(
        bytes.size(), static_cast<size_t>(std::numeric_limits<int>::max()));
    int transferred = 0;
    const int result = cyusb_bulk_transfer(
        handle_, config_.bulkOutEndpoint,
        const_cast<unsigned char *>(bytes.data()), static_cast<int>(request),
        &transferred, config_.transferTimeoutMs);
    if (transferred < 0 || static_cast<size_t>(transferred) > request)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "cyusb_bulk_transfer OUT returned invalid length %d; requested=%zu",
          transferred, request);
    if (result != 0)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "cyusb_bulk_transfer OUT: %s (%d); requested=%zu transferred=%d; "
          "failed prefix is not retried",
          libusb_error_name(result), result, request, transferred);
    return static_cast<size_t>(transferred);
  }

  llvm::Expected<size_t> read(llvm::MutableArrayRef<uint8_t> bytes) override {
    if (bytes.empty())
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "CIM USB read buffer cannot be empty");
    if (llvm::Error error = checkOpen())
      return std::move(error);

    std::lock_guard<std::mutex> transferLock(transferMutex_);
    if (llvm::Error error = checkOpen())
      return std::move(error);
    const size_t request = std::min(
        bytes.size(), static_cast<size_t>(std::numeric_limits<int>::max()));
    int transferred = 0;
    const int result = cyusb_bulk_transfer(
        handle_, config_.bulkInEndpoint, bytes.data(),
        static_cast<int>(request), &transferred, config_.transferTimeoutMs);
    if (transferred < 0 || static_cast<size_t>(transferred) > request)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "cyusb_bulk_transfer IN returned invalid length %d; requested=%zu",
          transferred, request);
    if (result != 0)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "cyusb_bulk_transfer IN: %s (%d); requested=%zu transferred=%d",
          libusb_error_name(result), result, request, transferred);
    return static_cast<size_t>(transferred);
  }

  void cancel() override {
    std::lock_guard<std::mutex> lock(stateMutex_);
    stopping_ = true;
  }

  void close() override {
    cancel();
    std::lock_guard<std::mutex> transferLock(transferMutex_);
    std::lock_guard<std::mutex> stateLock(stateMutex_);
    if (!handle_) {
      opened_ = false;
      if (sessionLock_.owns_lock())
        sessionLock_.unlock();
      return;
    }
    if (claimed_)
      cyusb_release_interface(handle_, config_.interfaceNumber);
    if (detached_)
      cyusb_attach_kernel_driver(handle_, config_.interfaceNumber);
    cyusb_close();
    handle_ = nullptr;
    claimed_ = false;
    detached_ = false;
    opened_ = false;
    if (sessionLock_.owns_lock())
      sessionLock_.unlock();
  }

private:
  llvm::Error checkOpen() const {
    std::lock_guard<std::mutex> lock(stateMutex_);
    if (!opened_ || !handle_ || stopping_)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "CIM USB device is not open");
    return llvm::Error::success();
  }

  llvm::Error validateInterface() const {
    libusb_config_descriptor *descriptor = nullptr;
    const int result = cyusb_get_active_config_descriptor(handle_, &descriptor);
    if (result != 0)
      return cyusbError("cyusb_get_active_config_descriptor", result);

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
    if (!interface || !foundIn || !foundOut)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "CIM USB interface/alternate setting "
                                     "lacks configured bulk endpoints");
    return llvm::Error::success();
  }

  UsbCIMDeviceConfig config_;
  cyusb_handle *handle_ = nullptr;
  bool claimed_ = false;
  bool detached_ = false;
  bool opened_ = false;
  bool stopping_ = true;
  std::unique_lock<std::mutex> sessionLock_;
  mutable std::mutex stateMutex_;
  std::mutex transferMutex_;
};

} // namespace

std::unique_ptr<UsbDriver> createPlatformUsbDriver() {
  return std::make_unique<LibcyusbUsbDriver>();
}

} // namespace cim22::runtime::detail
