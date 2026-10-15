//===- LibusbUsbDriver.cpp - Linux libusb driver ------------*- C++ -*-===//

#include "CIM22/Runtime/Detail/UsbDriver.h"

#include "llvm/Support/Error.h"

#include <libusb.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <string>

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

    if (libusb_kernel_driver_active(handle_, config.interfaceNumber) == 1) {
      result = libusb_detach_kernel_driver(handle_, config.interfaceNumber);
      if (result != LIBUSB_SUCCESS) {
        close();
        return libusbError("libusb_detach_kernel_driver", result);
      }
      detached_ = true;
    }
    result = libusb_claim_interface(handle_, config.interfaceNumber);
    if (result != LIBUSB_SUCCESS) {
      close();
      return libusbError("libusb_claim_interface", result);
    }
    claimed_ = true;
    result = libusb_set_interface_alt_setting(handle_, config.interfaceNumber,
                                              config.alternateSetting);
    if (result != LIBUSB_SUCCESS) {
      close();
      return libusbError("libusb_set_interface_alt_setting", result);
    }
    return llvm::Error::success();
  }

  llvm::Expected<size_t> write(llvm::ArrayRef<uint8_t> bytes) override {
    if (bytes.empty())
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "CIM USB write cannot be empty");
    return transfer(config_.bulkOutEndpoint,
                    const_cast<unsigned char *>(bytes.data()), bytes.size(),
                    config_.transferTimeoutMs, "libusb_bulk_transfer OUT");
  }

  llvm::Expected<size_t> read(llvm::MutableArrayRef<uint8_t> bytes) override {
    if (bytes.empty())
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "CIM USB read buffer cannot be empty");
    auto result =
        transfer(config_.bulkInEndpoint, bytes.data(), bytes.size(),
                 config_.receivePollTimeoutMs, "libusb_bulk_transfer IN");
    return result;
  }

  void cancel() override {
    // Synchronous libusb transfers are bounded by receivePollTimeoutMs, so the
    // common receive thread exits without an unbounded driver wait.
  }

  void close() override {
    if (handle_) {
      if (claimed_)
        libusb_release_interface(handle_, config_.interfaceNumber);
      if (detached_)
        libusb_attach_kernel_driver(handle_, config_.interfaceNumber);
      libusb_close(handle_);
    }
    handle_ = nullptr;
    claimed_ = false;
    detached_ = false;
    if (context_)
      libusb_exit(context_);
    context_ = nullptr;
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

  llvm::Expected<size_t> transfer(uint8_t endpoint, unsigned char *data,
                                  size_t size, uint32_t timeout,
                                  const llvm::Twine &operation) {
    if (!handle_)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "CIM USB device is not open");
    const size_t request =
        std::min(size, static_cast<size_t>(std::numeric_limits<int>::max()));
    int transferred = 0;
    const int result =
        libusb_bulk_transfer(handle_, endpoint, data, static_cast<int>(request),
                             &transferred, timeout);
    if (result == LIBUSB_ERROR_TIMEOUT)
      return static_cast<size_t>(transferred);
    if (result != LIBUSB_SUCCESS)
      return libusbError(operation, result);
    return static_cast<size_t>(transferred);
  }

  libusb_context *context_ = nullptr;
  libusb_device_handle *handle_ = nullptr;
  UsbCIMDeviceConfig config_;
  bool claimed_ = false;
  bool detached_ = false;
};
} // namespace

std::unique_ptr<UsbDriver> createPlatformUsbDriver() {
  return std::make_unique<LibusbUsbDriver>();
}
} // namespace cim22::runtime::detail
