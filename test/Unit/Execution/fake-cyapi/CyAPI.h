// Minimal SDK double for the Windows transport contract; no supplier code.
#pragma once

#include <algorithm>
#include <cassert>
#include <vector>
#ifndef _WINDOWS_
#error "Include windows.h before CyAPI.h, as required by the official SDK"
#endif

namespace fake_cyapi {
struct Device {
  USHORT vendor = 0x04b4, product = 0x00f1;
  UCHAR address = 1, interfaceNumber = 0;
};
inline std::vector<Device> devices;
inline OVERLAPPED *pending = nullptr;
inline PUCHAR buffer = nullptr;
inline LONG capacity = 0, completedLength = 0, writeLength = 3;
inline LONG requestedWrite = 0;
inline LONG abortTail = 0;
inline int begins = 0, finishes = 0, aborts = 0, closes = 0;
inline bool beginFails = false, finishFails = false, writeFails = false;
inline bool altFails = false, abortStalls = false, missingEndpoint = false;
inline bool cancelled = false;
inline UCHAR selected = 0;

inline void reset() {
  assert(!pending);
  devices = {Device{}};
  begins = finishes = aborts = closes = 0;
  beginFails = finishFails = writeFails = altFails = abortStalls = false;
  missingEndpoint = cancelled = false;
  writeLength = 3;
  abortTail = 0;
}
inline void complete(LONG length) {
  assert(pending);
  completedLength = length;
  std::fill_n(buffer, std::min(length, capacity), 0x5a);
  SetEvent(pending->hEvent);
}
} // namespace fake_cyapi

inline GUID CYUSBDRV_GUID{};
class CCyUSBEndPoint {
public:
  UCHAR Attributes = 2, Address = 0;
  bool bIn = false;
  ULONG TimeOut = 0, NtStatus = 0, UsbdStatus = 0;

  bool XferData(PUCHAR, LONG &length) {
    fake_cyapi::requestedWrite = length;
    length = fake_cyapi::writeLength;
    return !fake_cyapi::writeFails;
  }
  PUCHAR BeginDataXfer(PUCHAR bytes, LONG length, OVERLAPPED *overlapped) {
    using namespace fake_cyapi;
    assert(!pending);
    ++begins;
    if (beginFails)
      return nullptr;
    pending = overlapped;
    buffer = bytes;
    capacity = length;
    cancelled = false;
    return bytes;
  }
  bool FinishDataXfer(PUCHAR bytes, LONG &length, OVERLAPPED *overlapped,
                      PUCHAR context) {
    using namespace fake_cyapi;
    assert(pending == overlapped && bytes == buffer && context == bytes);
    ++finishes;
    pending = nullptr;
    length = completedLength;
    NtStatus = finishFails ? 0xc0000001u : cancelled ? 0xc0000120u : 0;
    UsbdStatus = finishFails ? 0xc0000002u : cancelled ? 0xc0010000u : 0;
    return !cancelled && !finishFails;
  }
  bool Abort() {
    using namespace fake_cyapi;
    ++aborts;
    assert(pending);
    if (abortStalls)
      return false;
    if (abortTail)
      complete(abortTail);
    else {
      cancelled = true;
      SetEvent(pending->hEvent);
    }
    return true;
  }
};
class CCyBulkEndPoint : public CCyUSBEndPoint {};

class CCyUSBDevice {
  CCyBulkEndPoint in_, out_;
  CCyUSBEndPoint *endpoints_[2] = {&in_, &out_};
  bool opened_ = false;

public:
  USHORT VendorID = 0, ProductID = 0;
  UCHAR USBAddress = 0;
  CCyUSBEndPoint **EndPoints = endpoints_;
  CCyUSBDevice(HANDLE = nullptr, GUID = CYUSBDRV_GUID, BOOL = true) {
    in_.Address = 0x81;
    in_.bIn = true;
    out_.Address = 0x01;
  }
  UCHAR DeviceCount() { return static_cast<UCHAR>(fake_cyapi::devices.size()); }
  bool Open(UCHAR index) {
    fake_cyapi::selected = index;
    const auto &d = fake_cyapi::devices.at(index);
    VendorID = d.vendor;
    ProductID = d.product;
    USBAddress = d.address;
    return opened_ = true;
  }
  void Close() {
    if (opened_)
      ++fake_cyapi::closes;
    opened_ = false;
  }
  UCHAR Interface() {
    return fake_cyapi::devices.at(fake_cyapi::selected).interfaceNumber;
  }
  bool SetAltIntfc(UCHAR) { return !fake_cyapi::altFails; }
  UCHAR EndPointCount() { return fake_cyapi::missingEndpoint ? 1 : 2; }
};
