//===- CyApiUsbDriver.cpp - Windows Cypress USB transport ---*- C++ -*-===//

#include "CIM22/Runtime/Detail/UsbDriver.h"

#include <windows.h>

// The official SDK requires Win32 types before its declarations.
#include <CyAPI.h>

#include <algorithm>
#include <atomic>
#include <memory>
#include <vector>

namespace cim22::runtime::detail {
namespace {
llvm::Error usbError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

llvm::Error transferError(const char *operation, CCyUSBEndPoint &endpoint) {
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "CyAPI %s failed: NTSTATUS=0x%08lx USBD_STATUS=0x%08lx", operation,
      endpoint.NtStatus, endpoint.UsbdStatus);
}

// The common runtime owns the receive thread. Keep one overlapped IN request
// alive across polls, as in FrameHandler_USB_32b; a poll timeout is not an
// abort.
class CyApiUsbDriver final : public UsbDriver {
  struct ReadRequest {
    std::vector<uint8_t> bytes;
    OVERLAPPED overlapped{};
    PUCHAR context = nullptr;

    explicit ReadRequest(size_t size) : bytes(size) {
      overlapped.hEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    }
    ~ReadRequest() {
      if (overlapped.hEvent)
        CloseHandle(overlapped.hEvent);
    }
  };

public:
  ~CyApiUsbDriver() override { close(); }

  llvm::Error open(const UsbCIMDeviceConfig &config) override {
    close();
    if (quarantined_)
      return usbError("CyAPI session has an unretired IN request; restart the "
                      "process after resetting the board");
    // CyAPI exposes USBAddress, but no equivalent of libusb's bus number.
    if (config.busNumber)
      return usbError(
          "CyAPI does not support USB bus selection; omit busNumber");
    if (config.receiveChunkBytes == 0 ||
        config.receiveChunkBytes > 4 * 1024 * 1024)
      return usbError("CyAPI receiveChunkBytes must be between 1 and 4 MiB");
    config_ = config;
    cancelled_ = false;
    device_ = std::make_unique<CCyUSBDevice>(nullptr, CYUSBDRV_GUID, false);
    unsigned matches = 0;
    UCHAR selected = 0;
    const unsigned count = device_->DeviceCount();
    for (unsigned index = 0; index < count; ++index) {
      if (!device_->Open(static_cast<UCHAR>(index)))
        continue;
      if (matchesProfile()) {
        selected = static_cast<UCHAR>(index);
        ++matches;
      }
      device_->Close();
    }
    if (matches != 1)
      return failOpen(matches == 0
                          ? "CyAPI CIM USB device not found"
                          : "CyAPI CIM USB device is ambiguous; specify "
                            "deviceAddress or connect only one board");
    if (!device_->Open(selected) || !matchesProfile())
      return failOpen("CyAPI device changed or could not be reopened");
    if (device_->Interface() != config.interfaceNumber ||
        !device_->SetAltIntfc(config.alternateSetting))
      return failOpen("CyAPI interface/alternate setting is unavailable");
    for (unsigned index = 0; index < device_->EndPointCount(); ++index) {
      auto *endpoint = device_->EndPoints[index];
      if (!endpoint || (endpoint->Attributes & 3) != 2)
        continue;
      if (endpoint->bIn && endpoint->Address == config.bulkInEndpoint)
        in_ = static_cast<CCyBulkEndPoint *>(endpoint);
      if (!endpoint->bIn && endpoint->Address == config.bulkOutEndpoint)
        out_ = static_cast<CCyBulkEndPoint *>(endpoint);
    }
    if (!in_ || !out_)
      return failOpen("CyAPI configured bulk endpoints are unavailable");
    out_->TimeOut = config.transferTimeoutMs;
    request_ = std::make_unique<ReadRequest>(config.receiveChunkBytes);
    if (!request_->overlapped.hEvent)
      return failOpen("CyAPI IN CreateEvent failed");
    // Queue IN before open returns, so even startupDelayMs=0 cannot send first.
    if (auto error = beginRead()) {
      close();
      return error;
    }
    return llvm::Error::success();
  }

  llvm::Expected<size_t> write(llvm::ArrayRef<uint8_t> bytes) override {
    if (!out_ || cancelled_)
      return usbError("CyAPI device is not open or is stopping");
    if (bytes.empty())
      return usbError("CyAPI OUT buffer cannot be empty");
    // Match the supplied handler's chip-test path: OUT chunks are at most
    // 512 bytes. The byte stream may split a 12-byte record across transfers.
    const size_t size = std::min(bytes.size(), size_t{512});
    LONG length = static_cast<LONG>(size);
    if (!out_->XferData(const_cast<PUCHAR>(bytes.data()), length))
      return transferError("OUT XferData", *out_);
    if (length <= 0 || size_t(length) > size)
      return usbError("CyAPI OUT returned an invalid transfer length");
    // The common runtime retries the remainder of a successful short write.
    return size_t(length);
  }

  llvm::Expected<size_t> read(llvm::MutableArrayRef<uint8_t> bytes) override {
    if (!request_)
      return usbError("CyAPI device is not open");
    if (bytes.size() < request_->bytes.size())
      return usbError("CyAPI IN destination is smaller than receiveChunkBytes");
    if (cancelled_)
      return 0;
    if (!request_->context)
      if (auto error = beginRead())
        return std::move(error);
    const DWORD result = WaitForSingleObject(request_->overlapped.hEvent,
                                             config_.receivePollTimeoutMs);
    if (result == WAIT_TIMEOUT)
      return 0;
    if (result != WAIT_OBJECT_0)
      return usbError("CyAPI IN event wait failed");
    return completeRead(bytes);
  }

  void cancel() override { cancelled_ = true; }

  llvm::Expected<size_t> drain(llvm::MutableArrayRef<uint8_t> bytes) override {
    if (!request_ || !request_->context)
      return size_t{0};
    if (bytes.size() < request_->bytes.size())
      return usbError("CyAPI IN destination is smaller than receiveChunkBytes");
    DWORD result = WaitForSingleObject(request_->overlapped.hEvent, 0);
    if (result != WAIT_OBJECT_0) {
      in_->Abort();
      result = WaitForSingleObject(request_->overlapped.hEvent, 2000);
    }
    if (result != WAIT_OBJECT_0) {
      // The driver still owns this storage. Retain it and its endpoint rather
      // than freeing an active OVERLAPPED. Ordinary stops never enter here.
      (void)request_.release();
      (void)device_.release();
      in_ = out_ = nullptr;
      quarantined_ = true;
      return usbError("CyAPI IN Abort did not return buffer ownership");
    }
    return completeRead(bytes);
  }

  // Called only after the common runtime has joined its receive thread.
  void close() override {
    cancelled_ = true;
    if (request_ && request_->context) {
      // Fallback for failed open/destruction. Normal shutdown calls drain
      // first and delivers its bytes/errors to the common receive queue.
      std::vector<uint8_t> tail(request_->bytes.size());
      auto result = drain(tail);
      if (!result)
        llvm::consumeError(result.takeError());
    }
    request_.reset();
    if (device_)
      device_->Close();
    device_.reset();
    in_ = out_ = nullptr;
  }

private:
  bool matchesProfile() const {
    return device_->VendorID == config_.vendorId &&
           device_->ProductID == config_.productId &&
           (!config_.deviceAddress ||
            device_->USBAddress == *config_.deviceAddress);
  }
  llvm::Error failOpen(const char *message) {
    close();
    return usbError(message);
  }
  llvm::Error beginRead() {
    ResetEvent(request_->overlapped.hEvent);
    request_->context = in_->BeginDataXfer(
        request_->bytes.data(), static_cast<LONG>(request_->bytes.size()),
        &request_->overlapped);
    if (!request_->context)
      return transferError("IN BeginDataXfer", *in_);
    return llvm::Error::success();
  }
  llvm::Expected<size_t> completeRead(llvm::MutableArrayRef<uint8_t> bytes) {
    LONG length = static_cast<LONG>(request_->bytes.size());
    const bool success =
        in_->FinishDataXfer(request_->bytes.data(), length,
                            &request_->overlapped, request_->context);
    request_->context = nullptr;
    if (!success) {
      // Match the supplier: only an expected cancellation during stop is
      // benign; other status codes still invalidate the receive transaction.
      if (cancelled_ &&
          (in_->NtStatus == 0xc0000120u || in_->UsbdStatus == 0xc0010000u))
        return size_t{0};
      return transferError("IN FinishDataXfer", *in_);
    }
    if (length < 0 || size_t(length) > request_->bytes.size())
      return usbError("CyAPI IN returned an invalid transfer length");
    std::copy_n(request_->bytes.data(), length, bytes.data());
    return size_t(length);
  }

  UsbCIMDeviceConfig config_;
  std::unique_ptr<CCyUSBDevice> device_;
  CCyBulkEndPoint *in_ = nullptr, *out_ = nullptr;
  std::unique_ptr<ReadRequest> request_;
  std::atomic<bool> cancelled_{true};
  bool quarantined_ = false;
};
} // namespace

std::unique_ptr<UsbDriver> createPlatformUsbDriver() {
  return std::make_unique<CyApiUsbDriver>();
}
} // namespace cim22::runtime::detail
