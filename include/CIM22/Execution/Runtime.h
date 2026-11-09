// Native runtime ABI for caller-owned, contiguous CPU tensors. The model
// signature supplies each buffer's element type. Keep the session alive and
// the buffers valid for the duration of every call.
#ifndef CIM22_EXECUTION_RUNTIME_H
#define CIM22_EXECUTION_RUNTIME_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct CIM22UsbConfig {
  uint16_t vendor_id;
  uint16_t product_id;
  // Use 0xff to omit a selector. Bus selection is Linux-only.
  uint8_t bus_number;
  uint8_t device_address;
  uint8_t interface_number;
  uint8_t alternate_setting;
  uint8_t bulk_in_endpoint;
  uint8_t bulk_out_endpoint;
  // Outbound record selectors, [0, 3] for link and [0, 7] for channel.
  uint8_t physical_link;
  uint8_t logical_channel;
  // USB record word order: 0 for big-endian, 1 for little-endian.
  uint8_t word_byte_order;
  // Per-transfer timeout, receive polling interval, and response deadline
  // per Macro batch after its requests have been sent, all in milliseconds.
  uint32_t transfer_timeout_ms;
  uint32_t receive_poll_timeout_ms;
  uint32_t response_timeout_ms;
} CIM22UsbConfig;
// Separate extension with packed 24-bit PLL words in phase order.
typedef struct CIM22PLLConfig {
  const char *port;
  uint32_t configuration_word;
  uint32_t work_word;
  uint32_t readback_word;
} CIM22PLLConfig;
// Board pacing in milliseconds. Zero skips a delay. Start-up gives the
// receive path time to settle. The remaining delays follow the completed
// operation. The Work and selection delays each cover an entire batch.
typedef struct CIM22HardwareConfig {
  uint32_t startup_delay_ms;
  uint32_t cache_write_delay_ms;
  uint32_t weight_write_delay_ms;
  uint32_t work_delay_ms;
  uint32_t readback_select_delay_ms;
} CIM22HardwareConfig;
// Shared models need only the execution runtime. Packet loading and object
// emission are compiler-only functions in libcim22_native_compiler.
// Both loaders return a session owned by the caller, or NULL on failure.
// Load, compile and configure calls accept an optional error buffer; its
// capacity is in bytes and includes the terminating NUL.
void *cim22_session_load_shared(const char *model_file, char *error,
                                size_t error_capacity);
void *cim22_session_load_packet(const char *packet_file, const char *entry,
                                char *error, size_t error_capacity);
// Native backend is "simulator" or "usb". BF16 semantics come from the IR.
int cim22_native_compile_object(const char *packet_file, const char *object_file,
                         const char *backend, char *error,
                         size_t error_capacity);
void cim22_session_destroy(void *session);
// Results follow inputs in tensor metadata. Returns -1 for an invalid index.
int64_t cim22_session_num_inputs(void *session);
int64_t cim22_session_num_outputs(void *session);
int64_t cim22_session_rank(void *session, int64_t tensor);
int64_t cim22_session_dim(void *session, int64_t tensor, int64_t axis);
// Artifact DataType value: INT8=1, FP32=3, INT32=4; -1 for an invalid index.
int64_t cim22_session_dtype(void *session, int64_t tensor);
int64_t cim22_session_num_transactions(void *session);
// Counts CIM transaction dispatch attempts during the most recent invoke.
uint64_t cim22_session_dispatch_count(void *session);
// Pass inputs followed by outputs in signature order. Counts are elements
// per buffer. Zero indicates success. On failure, fetch the diagnostic with
// cim22_session_error(); output contents are unspecified.
int cim22_session_invoke(void *session, void **buffers, const int64_t *counts,
                         int64_t num_buffers);
// Open or replace the USB device for a USB model using default pacing.
// Returns nonzero on configuration or transport failure.
int cim22_session_configure_usb(void *session, const CIM22UsbConfig *config,
                                char *error, size_t error_capacity);
// A null PLL config or null/empty port selects a manually managed clock.
// Linux also permits a missing port; Windows rejects nonempty serial ports.
int cim22_session_configure_usb_with_pll(void *session,
                                         const CIM22UsbConfig *config,
                                         const CIM22PLLConfig *pll, char *error,
                                         size_t error_capacity);
// Uses the supplied pacing configuration, copied during configuration.
int cim22_session_configure_usb_with_hardware(
    void *session, const CIM22UsbConfig *config, const CIM22PLLConfig *pll,
    const CIM22HardwareConfig *hardware, char *error, size_t error_capacity);
int64_t cim22_session_num_readbacks(void *session, int64_t transaction);
// Capture one USB transaction with BF16 input bits in binding order (64 per
// input). Returns one header and six raw body flits per readback, in binding
// order. Capture bypasses Host IR, numeric restoration and weight reuse.
// Counts are elements of the respective uint16_t and uint64_t arrays.
int cim22_session_capture_usb(void *session, int64_t transaction,
                              const uint16_t *inputs, size_t input_count,
                              uint64_t *flits, size_t flit_count);
// Borrowed error string. Copy it before another call can update the error or
// before destroying the session.
const char *cim22_session_error(void *session);
#ifdef __cplusplus
}
#endif
#endif
