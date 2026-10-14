// Native software-runtime ABI. Buffers are contiguous FP32, caller-owned.
// A session must not be destroyed while it is being invoked or inspected.
#ifndef CIM22_EXECUTION_RUNTIME_H
#define CIM22_EXECUTION_RUNTIME_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Shared models need only the execution runtime. Packet loading and object
// emission are compiler-only functions in libcim22_compiler.
void *cim22_session_load_shared(const char *model_file, char *error,
                                size_t error_capacity);
void *cim22_session_load_packet(const char *packet_file, const char *entry,
                                char *error, size_t error_capacity);
int cim22_compile_object(const char *packet_file, const char *object_file,
                         const void *output_schema, size_t schema_size,
                         char *error, size_t error_capacity);
const void *cim22_session_output_schema(void *session, size_t *size);
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
const char *cim22_session_error(void *session);
#ifdef __cplusplus
}
#endif
#endif
