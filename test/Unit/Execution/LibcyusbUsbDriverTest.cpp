//===- LibcyusbUsbDriverTest.cpp - CyUSB driver tests --------*- C++ -*-===//

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>

#include <cyusb.h>

namespace {
struct Fake {
  cyusb_handle *handle = reinterpret_cast<cyusb_handle *>(0x1);
  int bus = 4;
  int address = 1;
  int alternate = 0;
  int activeDriver = 0;
  int openResult = 1;
  int setAlternateCalls = 0;
  int claimCalls = 0;
  int releaseCalls = 0;
  int detachCalls = 0;
  int attachCalls = 0;
  int closeCalls = 0;
  int bulkCalls = 0;
  int bulkResult = 0;
  std::vector<unsigned char> input{1, 2, 3};
} fake;

libusb_endpoint_descriptor endpoints[2]{};
libusb_interface_descriptor alternateSetting{};
libusb_interface interfaceEntry{};
libusb_interface interfaces[1]{};
libusb_config_descriptor configuration{};

void resetFake() {
  fake = Fake{};
  endpoints[0] = {};
  endpoints[0].bEndpointAddress = 0x86;
  endpoints[0].bmAttributes = LIBUSB_TRANSFER_TYPE_BULK;
  endpoints[1] = {};
  endpoints[1].bEndpointAddress = 0x02;
  endpoints[1].bmAttributes = LIBUSB_TRANSFER_TYPE_BULK;
  alternateSetting = {};
  alternateSetting.bInterfaceNumber = 0;
  alternateSetting.bAlternateSetting = 0;
  alternateSetting.bNumEndpoints = 2;
  alternateSetting.endpoint = endpoints;
  interfaceEntry = {};
  interfaceEntry.num_altsetting = 1;
  interfaceEntry.altsetting = &alternateSetting;
  interfaces[0] = interfaceEntry;
  configuration = {};
  configuration.bNumInterfaces = 1;
  configuration.interface = interfaces;
}
} // namespace

int cyusb_open(unsigned short, unsigned short) { return fake.openResult; }
cyusb_handle *cyusb_gethandle(int) { return fake.handle; }
int cyusb_get_busnumber(cyusb_handle *) { return fake.bus; }
int cyusb_get_devaddr(cyusb_handle *) { return fake.address; }
int cyusb_get_active_config_descriptor(cyusb_handle *,
                                       libusb_config_descriptor **out) {
  *out = &configuration;
  return 0;
}
void cyusb_free_config_descriptor(libusb_config_descriptor *) {}
int cyusb_kernel_driver_active(cyusb_handle *, int) {
  return fake.activeDriver;
}
int cyusb_detach_kernel_driver(cyusb_handle *, int) {
  ++fake.detachCalls;
  fake.activeDriver = 0;
  return 0;
}
int cyusb_attach_kernel_driver(cyusb_handle *, int) {
  ++fake.attachCalls;
  return 0;
}
int cyusb_claim_interface(cyusb_handle *, int) {
  ++fake.claimCalls;
  return 0;
}
int cyusb_release_interface(cyusb_handle *, int) {
  ++fake.releaseCalls;
  return 0;
}
int cyusb_set_interface_alt_setting(cyusb_handle *, int, int alternate) {
  ++fake.setAlternateCalls;
  fake.alternate = alternate;
  return 0;
}
void cyusb_close() { ++fake.closeCalls; }
int cyusb_bulk_transfer(cyusb_handle *, unsigned char endpoint,
                        unsigned char *data, int length, int *transferred,
                        int) {
  ++fake.bulkCalls;
  if (fake.bulkResult != 0) {
    *transferred = 0;
    return fake.bulkResult;
  }
  if ((endpoint & LIBUSB_ENDPOINT_DIR_MASK) == LIBUSB_ENDPOINT_IN) {
    const int count = std::min(length, static_cast<int>(fake.input.size()));
    std::memcpy(data, fake.input.data(), count);
    *transferred = count;
  } else {
    *transferred = length;
  }
  return 0;
}
int libusb_control_transfer(libusb_device_handle *, uint8_t, uint8_t, uint16_t,
                            uint16_t, unsigned char *data, uint16_t length,
                            unsigned int) {
  assert(length == 1);
  *data = static_cast<unsigned char>(fake.alternate);
  return 1;
}
const char *libusb_error_name(int code) {
  return code == LIBUSB_ERROR_IO ? "LIBUSB_ERROR_IO" : "LIBUSB_ERROR_OTHER";
}

#include "../../../runtime/LibcyusbUsbDriver.cpp"

using cim22::runtime::detail::createPlatformUsbDriver;

int main() {
  resetFake();
  cim22::runtime::UsbCIMDeviceConfig config;
  config.vendorId = 0x04b4;
  config.productId = 0x00f1;
  config.busNumber = 4;
  config.deviceAddress = 1;
  config.bulkInEndpoint = 0x86;
  config.bulkOutEndpoint = 0x02;
  config.transferTimeoutMs = 100;

  auto driver = createPlatformUsbDriver();
  assert(!driver->open(config));
  assert(fake.claimCalls == 1);
  assert(fake.setAlternateCalls == 0);
  unsigned char output[4]{};
  auto read = driver->read({output, sizeof(output)});
  assert(read && *read == 3 && output[0] == 1 && output[2] == 3);
  const unsigned char input[] = {4, 5};
  auto written = driver->write({input, sizeof(input)});
  assert(written && *written == sizeof(input));
  driver->close();
  assert(fake.releaseCalls == 1 && fake.closeCalls == 1);

  resetFake();
  fake.activeDriver = 1;
  driver = createPlatformUsbDriver();
  assert(!driver->open(config));
  assert(fake.detachCalls == 1);
  driver->close();
  assert(fake.attachCalls == 1);

  resetFake();
  fake.address = 9;
  driver = createPlatformUsbDriver();
  assert(driver->open(config));
  assert(fake.closeCalls == 1);
  return 0;
}
