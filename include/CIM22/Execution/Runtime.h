// Native runtime ABI. Inference buffers are contiguous FP32, caller-owned.
// A session must not be destroyed while it is being invoked or inspected.
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
  uint8_t bus_number;
  uint8_t device_address;
  uint8_t interface_number;
  uint8_t alternate_setting;
  uint8_t bulk_in_endpoint;
  uint8_t bulk_out_endpoint;
  uint8_t physical_link;
  uint8_t logical_channel;
  uint8_t word_byte_order;
  uint32_t transfer_timeout_ms;
  uint32_t receive_poll_timeout_ms;
  uint32_t response_timeout_ms;
} CIM22UsbConfig;
// Shared models need only the execution runtime. Packet loading and object
// emission are compiler-only functions in libcim22_compiler.
void *cim22_session_load_shared(const char *model_file, char *error,
                                size_t error_capacity);
void *cim22_session_load_packet(const char *packet_file, const char *entry,
                                char *error, size_t error_capacity);
// backend must be non-null and name either "software" or "usb".
int cim22_compile_object(const char *packet_file, const char *object_file,
                         const char *backend, char *error,
                         size_t error_capacity);
void cim22_session_destroy(void *session);
// Results follow inputs in tensor metadata. Returns -1 for an invalid index.
int64_t cim22_session_num_inputs(void *session);
int64_t cim22_session_num_outputs(void *session);
int64_t cim22_session_rank(void *session, int64_t tensor);
int64_t cim22_session_dim(void *session, int64_t tensor, int64_t axis);
int64_t cim22_session_num_transactions(void *session);
uint64_t cim22_session_dispatch_count(void *session);
// Counts are numbers of FP32 elements. Failure returns nonzero, with diagnostic
// available until the next invocation. Failed output buffers are unspecified.
int cim22_session_invoke(void *session, void **buffers, const int64_t *counts,
                         int64_t num_buffers);
int cim22_session_configure_usb(void *session, const CIM22UsbConfig *config,
                                char *error, size_t error_capacity);
int64_t cim22_session_num_readbacks(void *session, int64_t transaction);
// Diagnostic only: BF16 input bits in dynamic-binding order (64 per row).
// Output is one header and six body flits per readback, in binding order.
// Does not execute Host IR or decode tensor values. Counts are element counts.
int cim22_session_capture_usb(void *session, int64_t transaction,
                              const uint16_t *inputs, size_t input_count,
                              uint64_t *flits, size_t flit_count);
const char *cim22_session_error(void *session);
#ifdef __cplusplus
}
#endif
#endif
