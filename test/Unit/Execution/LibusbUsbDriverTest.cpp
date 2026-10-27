//===- LibusbUsbDriverTest.cpp - Platform libusb lifecycle tests
//-----------===//
// No libusb library is linked: the C API below is a deterministic fake.

#include "UsbDriver.h"

#include "llvm/Support/Error.h"

#include <libusb.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <future>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

namespace {
struct Completion {
  libusb_transfer_status status = LIBUSB_TRANSFER_COMPLETED;
  int length = 0;
  std::vector<uint8_t> bytes;
};

struct FakeLibusb {
  std::mutex mutex;
  std::condition_variable condition;
  libusb_transfer *active = nullptr;
  std::deque<Completion> completions;
  std::vector<std::string> events;
  std::vector<uint8_t> cancelBytes;
  int submits = 0;
  int callbacks = 0;
  int cancels = 0;
  int frees = 0;
  int closes = 0;
  int releases = 0;
  int detachCalls = 0;
  int attachCalls = 0;
  int bulkCalls = 0;
  int getInterfaceCalls = 0;
  int setInterfaceCalls = 0;
  int getInterfaceResult = 1;
  int setInterfaceResult = LIBUSB_SUCCESS;
  uint8_t currentAlternate = 0;
  int eventCalls = 0;
  int idleEvents = 0;
  int submitFailOn = 0;
  int eventError = 0;
  int bulkResult = LIBUSB_SUCCESS;
  int bulkTransferred = -2;
  bool dispatchOnWrite = false;
  bool eventEntered = false;
  bool suppressEventDispatch = false;

  void reset() {
    std::lock_guard<std::mutex> lock(mutex);
    assert(!active);
    completions.clear();
    events.clear();
    cancelBytes.clear();
    submits = callbacks = cancels = frees = closes = releases = 0;
    detachCalls = attachCalls = bulkCalls = eventCalls = idleEvents = 0;
    getInterfaceCalls = setInterfaceCalls = 0;
    getInterfaceResult = 1;
    setInterfaceResult = LIBUSB_SUCCESS;
    currentAlternate = 0;
    submitFailOn = eventError = 0;
    bulkResult = LIBUSB_SUCCESS;
    bulkTransferred = -2;
    dispatchOnWrite = eventEntered = suppressEventDispatch = false;
  }

  void queue(Completion completion) {
    std::lock_guard<std::mutex> lock(mutex);
    completions.push_back(std::move(completion));
    condition.notify_all();
  }

  bool dispatch() {
    libusb_transfer *transfer;
    Completion completion;
    {
      std::lock_guard<std::mutex> lock(mutex);
      if (!active || completions.empty())
        return false;
      transfer = active;
      active = nullptr;
      completion = std::move(completions.front());
      completions.pop_front();
      ++callbacks;
      events.push_back("callback");
    }
    transfer->status = completion.status;
    transfer->actual_length = completion.length;
    std::copy_n(completion.bytes.begin(),
                std::min(completion.bytes.size(),
                         static_cast<size_t>(transfer->length)),
                transfer->buffer);
    transfer->callback(transfer);
    condition.notify_all();
    return true;
  }
};

FakeLibusb fake;
int deviceToken;
int handleToken;
int contextToken;
libusb_device *fakeDevice = reinterpret_cast<libusb_device *>(&deviceToken);
libusb_device_handle *fakeHandle =
    reinterpret_cast<libusb_device_handle *>(&handleToken);
libusb_context *fakeContext = reinterpret_cast<libusb_context *>(&contextToken);
libusb_device *deviceList[] = {fakeDevice, nullptr};
libusb_endpoint_descriptor endpoints[2]{};
libusb_interface_descriptor alternate{};
libusb_interface interface{};
libusb_config_descriptor configuration{};

void prepareDescriptors() {
  endpoints[0].bEndpointAddress = 0x81;
  endpoints[1].bEndpointAddress = 0x01;
  for (auto &endpoint : endpoints) {
    endpoint.bmAttributes = LIBUSB_TRANSFER_TYPE_BULK;
    endpoint.wMaxPacketSize = 1024;
  }
  alternate.bInterfaceNumber = 0;
  alternate.bAlternateSetting = 0;
  alternate.bNumEndpoints = 2;
  alternate.endpoint = endpoints;
  interface.altsetting = &alternate;
  interface.num_altsetting = 1;
  configuration.bNumInterfaces = 1;
  configuration.interface = &interface;
}
} // namespace

extern "C" {
int LIBUSB_CALL libusb_init(libusb_context **context) {
  *context = fakeContext;
  return LIBUSB_SUCCESS;
}
void LIBUSB_CALL libusb_exit(libusb_context *) {}
ssize_t LIBUSB_CALL libusb_get_device_list(libusb_context *,
                                           libusb_device ***devices) {
  *devices = deviceList;
  return 1;
}
void LIBUSB_CALL libusb_free_device_list(libusb_device **, int) {}
int LIBUSB_CALL libusb_get_device_descriptor(libusb_device *,
                                             libusb_device_descriptor *out) {
  *out = {};
  out->idVendor = 0x04b4;
  out->idProduct = 0x00f1;
  return LIBUSB_SUCCESS;
}
uint8_t LIBUSB_CALL libusb_get_bus_number(libusb_device *) { return 4; }
uint8_t LIBUSB_CALL libusb_get_device_address(libusb_device *) { return 1; }
int LIBUSB_CALL libusb_open(libusb_device *, libusb_device_handle **handle) {
  *handle = fakeHandle;
  return LIBUSB_SUCCESS;
}
void LIBUSB_CALL libusb_close(libusb_device_handle *) {
  std::lock_guard<std::mutex> lock(fake.mutex);
  assert(!fake.active);
  ++fake.closes;
  fake.events.push_back("close");
}
int LIBUSB_CALL libusb_get_active_config_descriptor(
    libusb_device *, libusb_config_descriptor **out) {
  *out = &configuration;
  return LIBUSB_SUCCESS;
}
void LIBUSB_CALL libusb_free_config_descriptor(libusb_config_descriptor *) {}
int LIBUSB_CALL libusb_claim_interface(libusb_device_handle *, int) {
  return LIBUSB_SUCCESS;
}
int LIBUSB_CALL libusb_release_interface(libusb_device_handle *, int) {
  ++fake.releases;
  return LIBUSB_SUCCESS;
}
int LIBUSB_CALL libusb_set_interface_alt_setting(libusb_device_handle *, int,
                                                 int alternate) {
  ++fake.setInterfaceCalls;
  if (fake.setInterfaceResult == LIBUSB_SUCCESS)
    fake.currentAlternate = static_cast<uint8_t>(alternate);
  return fake.setInterfaceResult;
}
int LIBUSB_CALL libusb_control_transfer(libusb_device_handle *, uint8_t type,
                                        uint8_t request, uint16_t value,
                                        uint16_t index, unsigned char *data,
                                        uint16_t length, unsigned int timeout) {
  assert(type == 0x81 && request == LIBUSB_REQUEST_GET_INTERFACE);
  assert(value == 0 && index == 0 && length == 1 && timeout == 5000);
  ++fake.getInterfaceCalls;
  *data = fake.currentAlternate;
  return fake.getInterfaceResult;
}
int LIBUSB_CALL libusb_kernel_driver_active(libusb_device_handle *, int) {
  return 1;
}
int LIBUSB_CALL libusb_detach_kernel_driver(libusb_device_handle *, int) {
  ++fake.detachCalls;
  return LIBUSB_SUCCESS;
}
int LIBUSB_CALL libusb_attach_kernel_driver(libusb_device_handle *, int) {
  ++fake.attachCalls;
  return LIBUSB_SUCCESS;
}
libusb_transfer *LIBUSB_CALL libusb_alloc_transfer(int) {
  return static_cast<libusb_transfer *>(
      std::calloc(1, sizeof(libusb_transfer)));
}
void LIBUSB_CALL libusb_free_transfer(libusb_transfer *transfer) {
  std::lock_guard<std::mutex> lock(fake.mutex);
  assert(fake.active != transfer);
  ++fake.frees;
  fake.events.push_back("free");
  std::free(transfer);
}
int LIBUSB_CALL libusb_submit_transfer(libusb_transfer *transfer) {
  std::lock_guard<std::mutex> lock(fake.mutex);
  ++fake.submits;
  fake.events.push_back("submit");
  assert(!fake.active);
  assert(transfer->endpoint == 0x81 && transfer->length == 32768);
  assert(transfer->timeout == 0);
  if (fake.submits == fake.submitFailOn)
    return LIBUSB_ERROR_IO;
  fake.active = transfer;
  return LIBUSB_SUCCESS;
}
int LIBUSB_CALL libusb_cancel_transfer(libusb_transfer *transfer) {
  std::lock_guard<std::mutex> lock(fake.mutex);
  ++fake.cancels;
  fake.events.push_back("cancel");
  if (fake.active != transfer || !fake.completions.empty())
    return LIBUSB_ERROR_NOT_FOUND;
  fake.completions.push_back({LIBUSB_TRANSFER_CANCELLED,
                              static_cast<int>(fake.cancelBytes.size()),
                              fake.cancelBytes});
  fake.condition.notify_all();
  return LIBUSB_SUCCESS;
}
void LIBUSB_CALL libusb_interrupt_event_handler(libusb_context *) {
  fake.condition.notify_all();
}
int LIBUSB_CALL libusb_handle_events_timeout_completed(libusb_context *,
                                                       timeval *timeout,
                                                       int *) {
  bool dispatch;
  {
    std::unique_lock<std::mutex> lock(fake.mutex);
    ++fake.eventCalls;
    fake.eventEntered = true;
    fake.condition.notify_all();
    assert(timeout->tv_sec == 0 && timeout->tv_usec == 200000);
    if (fake.eventError) {
      const int result = fake.eventError;
      fake.eventError = 0;
      return result;
    }
    if (fake.idleEvents > 0) {
      --fake.idleEvents;
      return LIBUSB_SUCCESS;
    }
    fake.condition.wait_for(lock, std::chrono::milliseconds(10),
                            [] { return !fake.completions.empty(); });
    dispatch = !fake.suppressEventDispatch;
  }
  if (dispatch)
    fake.dispatch();
  return LIBUSB_SUCCESS;
}
int LIBUSB_CALL libusb_bulk_transfer(libusb_device_handle *, unsigned char ep,
                                     unsigned char *, int length,
                                     int *transferred, unsigned int timeout) {
  bool dispatch;
  int result;
  {
    std::lock_guard<std::mutex> lock(fake.mutex);
    ++fake.bulkCalls;
    fake.events.push_back("write");
    assert(fake.submits > 0 && ep == 0x01 && timeout == 5000);
    *transferred = fake.bulkTransferred == -2 ? length : fake.bulkTransferred;
    result = fake.bulkResult;
    dispatch = fake.dispatchOnWrite;
  }
  if (dispatch)
    fake.dispatch();
  return result;
}
const char *LIBUSB_CALL libusb_error_name(int code) {
  switch (code) {
  case LIBUSB_ERROR_TIMEOUT:
    return "LIBUSB_ERROR_TIMEOUT";
  case LIBUSB_ERROR_IO:
    return "LIBUSB_ERROR_IO";
  case LIBUSB_ERROR_INTERRUPTED:
    return "LIBUSB_ERROR_INTERRUPTED";
  case LIBUSB_ERROR_NO_DEVICE:
    return "LIBUSB_ERROR_NO_DEVICE";
  default:
    return "LIBUSB_ERROR_OTHER";
  }
}
} // extern "C"

// Compile the actual platform implementation against only the fake C API.
#include "../../../lib/Runtime/LibusbUsbDriver.cpp"

namespace {
using cim22::runtime::UsbCIMDeviceConfig;
using cim22::runtime::detail::createPlatformUsbDriver;

UsbCIMDeviceConfig config() {
  UsbCIMDeviceConfig value;
  value.vendorId = 0x04b4;
  value.productId = 0x00f1;
  value.bulkInEndpoint = 0x81;
  value.bulkOutEndpoint = 0x01;
  return value;
}

void requireSuccess(llvm::Error error) {
  if (error) {
    std::cerr << llvm::toString(std::move(error)) << '\n';
    std::abort();
  }
}

std::string requireFailure(llvm::Expected<size_t> value) {
  assert(!value);
  return llvm::toString(value.takeError());
}

void checkCleanup() {
  assert(fake.frees == 1 && fake.closes == 1 && fake.releases == 1);
  assert(!fake.active && !fake.detachCalls && !fake.attachCalls);
  const auto free = std::find(fake.events.begin(), fake.events.end(), "free");
  const auto close = std::find(fake.events.begin(), fake.events.end(), "close");
  assert(free < close);
}

void testOpenAndClose() {
  fake.reset();
  auto driver = createPlatformUsbDriver();
  requireSuccess(driver->open(config()));
  assert(fake.submits == 1 && fake.active);
  std::array<uint8_t, 12> bytes{};
  auto count = driver->write(bytes);
  assert(count && *count == bytes.size());
  assert(fake.events[0] == "submit" && fake.events[1] == "write");
  driver->close();
  assert(fake.cancels == 1 && fake.callbacks == 1);
  checkCleanup();
}

void testAlternateSettingOnReopen() {
  fake.reset();
  auto driver = createPlatformUsbDriver();
  requireSuccess(driver->open(config()));
  driver->close();
  requireSuccess(driver->open(config()));
  driver->close();
  assert(fake.getInterfaceCalls == 2 && fake.setInterfaceCalls == 0);
  assert(fake.cancels == 2 && fake.callbacks == 2 && fake.releases == 2);

  fake.reset();
  libusb_interface_descriptor settings[2] = {alternate, alternate};
  settings[1].bAlternateSetting = 1;
  interface.altsetting = settings;
  interface.num_altsetting = 2;
  fake.currentAlternate = 1;
  requireSuccess(driver->open(config()));
  assert(fake.getInterfaceCalls == 1 && fake.setInterfaceCalls == 1);
  assert(fake.currentAlternate == 0);
  driver->close();
  checkCleanup();

  fake.reset();
  fake.currentAlternate = 1;
  fake.setInterfaceResult = LIBUSB_ERROR_IO;
  auto error = driver->open(config());
  assert(error && llvm::toString(std::move(error)).find("alt_setting") !=
                      std::string::npos);
  assert(fake.submits == 0 && fake.setInterfaceCalls == 1);
  assert(fake.releases == 1 && fake.closes == 1);
  prepareDescriptors();
}

void testGetInterfaceFailure() {
  for (int result : std::array<int, 3>{LIBUSB_ERROR_IO, 0, 2}) {
    fake.reset();
    fake.getInterfaceResult = result;
    auto driver = createPlatformUsbDriver();
    auto error = driver->open(config());
    assert(error && llvm::toString(std::move(error)).find("GET_INTERFACE") !=
                        std::string::npos);
    assert(fake.getInterfaceCalls == 1 && fake.setInterfaceCalls == 0);
    assert(fake.submits == 0 && fake.frees == 0);
    assert(fake.releases == 1 && fake.closes == 1);
  }
}

void testSubmitFailure() {
  fake.reset();
  fake.submitFailOn = 1;
  auto driver = createPlatformUsbDriver();
  llvm::Error error = driver->open(config());
  assert(error);
  assert(llvm::toString(std::move(error)).find("submit_transfer") !=
         std::string::npos);
  assert(fake.submits == 1 && fake.cancels == 0 && fake.callbacks == 0);
  checkCleanup();
}

void testIdleAndResubmit() {
  fake.reset();
  fake.idleEvents = 4;
  auto driver = createPlatformUsbDriver();
  requireSuccess(driver->open(config()));
  fake.queue({LIBUSB_TRANSFER_COMPLETED, 3, {1, 2, 3}});
  std::array<uint8_t, 32768> bytes{};
  auto count = driver->read(bytes);
  assert(count && *count == 3);
  assert(bytes[0] == 1 && bytes[1] == 2 && bytes[2] == 3);
  assert(fake.eventCalls == 5 && fake.submits == 2 && fake.cancels == 0);
  fake.queue({LIBUSB_TRANSFER_COMPLETED, 0, {}});
  count = driver->read(bytes);
  assert(count && *count == 0 && fake.submits == 3);
  driver->close();
  checkCleanup();
}

void testResubmitFailure() {
  fake.reset();
  fake.submitFailOn = 2;
  auto driver = createPlatformUsbDriver();
  requireSuccess(driver->open(config()));
  fake.queue({LIBUSB_TRANSFER_COMPLETED, 3, {1, 2, 3}});
  std::array<uint8_t, 32768> bytes{};
  assert(requireFailure(driver->read(bytes)).find("submit_transfer") !=
         std::string::npos);
  assert(fake.submits == 2 && fake.cancels == 0);
  driver->close();
  checkCleanup();
}

void testCancelAndPartialData() {
  fake.reset();
  auto driver = createPlatformUsbDriver();
  requireSuccess(driver->open(config()));
  fake.cancelBytes = {9, 8};
  driver->cancel();
  std::array<uint8_t, 32768> bytes{};
  auto count = driver->read(bytes);
  assert(count && *count == 2 && bytes[0] == 9 && bytes[1] == 8);
  count = driver->read(bytes);
  assert(count && *count == 0 && fake.submits == 1);
  driver->close();
  assert(fake.cancels == 1);
  checkCleanup();
}

void testCancelWaitingReader() {
  fake.reset();
  auto driver = createPlatformUsbDriver();
  requireSuccess(driver->open(config()));
  auto reader = std::async(std::launch::async, [&] {
    std::array<uint8_t, 32768> bytes{};
    return driver->read(bytes);
  });
  {
    std::unique_lock<std::mutex> lock(fake.mutex);
    assert(fake.condition.wait_for(lock, std::chrono::seconds(1),
                                   [] { return fake.eventEntered; }));
  }
  driver->cancel();
  assert(reader.wait_for(std::chrono::seconds(1)) == std::future_status::ready);
  auto count = reader.get();
  assert(count && *count == 0);
  driver->close();
  checkCleanup();
}

void testCallbackOnWriterWhileReaderWaits() {
  fake.reset();
  auto driver = createPlatformUsbDriver();
  requireSuccess(driver->open(config()));
  fake.dispatchOnWrite = true;
  fake.suppressEventDispatch = true;
  std::array<uint8_t, 32768> bytes{};
  auto reader =
      std::async(std::launch::async, [&] { return driver->read(bytes); });
  {
    std::unique_lock<std::mutex> lock(fake.mutex);
    assert(fake.condition.wait_for(lock, std::chrono::seconds(1),
                                   [] { return fake.eventEntered; }));
  }
  fake.queue({LIBUSB_TRANSFER_COMPLETED, 3, {4, 5, 6}});
  std::array<uint8_t, 12> out{};
  auto written = driver->write(out);
  assert(written && *written == out.size());
  assert(reader.wait_for(std::chrono::seconds(1)) == std::future_status::ready);
  auto count = reader.get();
  assert(count && *count == 3);
  assert(bytes[0] == 4 && bytes[1] == 5 && bytes[2] == 6);
  assert(fake.submits == 2 && fake.bulkCalls == 1);
  fake.suppressEventDispatch = false;
  driver->close();
  checkCleanup();
}

void testCompletedDataAfterCancel() {
  fake.reset();
  auto driver = createPlatformUsbDriver();
  requireSuccess(driver->open(config()));
  fake.queue({LIBUSB_TRANSFER_COMPLETED, 2, {7, 8}});
  driver->cancel();
  std::array<uint8_t, 32768> bytes{};
  auto count = driver->read(bytes);
  assert(count && *count == 2 && bytes[0] == 7 && bytes[1] == 8);
  count = driver->read(bytes);
  assert(count && *count == 0 && fake.submits == 1);
  driver->close();
  assert(fake.cancels == 1);
  checkCleanup();
}

void testReadErrorsAndLengths() {
  for (Completion completion :
       {Completion{LIBUSB_TRANSFER_ERROR, 0, {}},
        Completion{LIBUSB_TRANSFER_STALL, 2, {1, 2}},
        Completion{LIBUSB_TRANSFER_NO_DEVICE, 0, {}},
        Completion{LIBUSB_TRANSFER_TIMED_OUT, 0, {}},
        Completion{LIBUSB_TRANSFER_OVERFLOW, 0, {}},
        Completion{LIBUSB_TRANSFER_CANCELLED, 0, {}},
        Completion{LIBUSB_TRANSFER_COMPLETED, -1, {}},
        Completion{LIBUSB_TRANSFER_COMPLETED, 32769, {}}}) {
    fake.reset();
    auto driver = createPlatformUsbDriver();
    requireSuccess(driver->open(config()));
    fake.queue(completion);
    std::array<uint8_t, 32768> bytes{};
    auto error = requireFailure(driver->read(bytes));
    assert(!error.empty());
    assert(!requireFailure(driver->write(bytes)).empty());
    assert(fake.submits == 1 && fake.bulkCalls == 0);
    driver->close();
    checkCleanup();
  }
  fake.reset();
  auto driver = createPlatformUsbDriver();
  requireSuccess(driver->open(config()));
  fake.queue({LIBUSB_TRANSFER_COMPLETED, 3, {1, 2, 3}});
  std::array<uint8_t, 2> small{};
  assert(requireFailure(driver->read(small)).find("caller buffer") !=
         std::string::npos);
  driver->close();
  checkCleanup();
}

void testOutFailureAndPartialSuccess() {
  for (int code :
       {LIBUSB_ERROR_TIMEOUT, LIBUSB_ERROR_IO, LIBUSB_ERROR_NO_DEVICE}) {
    for (int transferred : {0, 5}) {
      fake.reset();
      auto driver = createPlatformUsbDriver();
      requireSuccess(driver->open(config()));
      fake.bulkResult = code;
      fake.bulkTransferred = transferred;
      std::array<uint8_t, 12> bytes{};
      const auto error = requireFailure(driver->write(bytes));
      assert(error.find("requested=12") != std::string::npos);
      assert(error.find("transferred=" + std::to_string(transferred)) !=
             std::string::npos);
      assert(error.find(libusb_error_name(code)) != std::string::npos);
      assert(fake.bulkCalls == 1);
      driver->close();
      checkCleanup();
    }
  }
  fake.reset();
  auto driver = createPlatformUsbDriver();
  requireSuccess(driver->open(config()));
  std::array<uint8_t, 12> bytes{};
  fake.bulkTransferred = 5;
  auto count = driver->write(bytes);
  assert(count && *count == 5);
  for (int invalid : {-1, 13}) {
    fake.bulkTransferred = invalid;
    assert(requireFailure(driver->write(bytes)).find("invalid length") !=
           std::string::npos);
  }
  driver->close();
  checkCleanup();
}

void testCallbackDuringOutAndEventsError() {
  fake.reset();
  auto driver = createPlatformUsbDriver();
  requireSuccess(driver->open(config()));
  fake.queue({LIBUSB_TRANSFER_ERROR, 0, {}});
  fake.dispatchOnWrite = true;
  std::array<uint8_t, 12> bytes{};
  assert(requireFailure(driver->write(bytes)).find("IN status") !=
         std::string::npos);
  assert(fake.bulkCalls == 1);
  assert(requireFailure(driver->write(bytes)).find("IN status") !=
         std::string::npos);
  assert(fake.bulkCalls == 1);
  driver->close();
  checkCleanup();

  fake.reset();
  driver = createPlatformUsbDriver();
  requireSuccess(driver->open(config()));
  fake.eventError = LIBUSB_ERROR_IO;
  std::array<uint8_t, 32768> buffer{};
  assert(requireFailure(driver->read(buffer)).find("handle_events") !=
         std::string::npos);
  driver->close();
  checkCleanup();
}
} // namespace

int main() {
  prepareDescriptors();
  testOpenAndClose();
  testAlternateSettingOnReopen();
  testGetInterfaceFailure();
  testSubmitFailure();
  testIdleAndResubmit();
  testResubmitFailure();
  testCancelAndPartialData();
  testCancelWaitingReader();
  testCallbackOnWriterWhileReaderWaits();
  testCompletedDataAfterCancel();
  testReadErrorsAndLengths();
  testOutFailureAndPartialSuccess();
  testCallbackDuringOutAndEventsError();
  std::cout
      << "PASS: fake-libusb startup, persistent IN, callbacks, cancellation, "
         "lengths and OUT errors\n";
}
