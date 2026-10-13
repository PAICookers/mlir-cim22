// Linux model-library ABI. This descriptor and its data live in the model DSO.
#ifndef CIM22_EXECUTION_MODELABI_H
#define CIM22_EXECUTION_MODELABI_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct CIM22ModelV1 {
  uint64_t abi_version;
  const uint8_t *data;
  uint64_t data_size;
  void (*invoke)(void **);
} CIM22ModelV1;
// Exported by each compiled model; no compiler/JIT is called by this function.
const CIM22ModelV1 *cim22_model_v1(void);
#ifdef __cplusplus
}
#endif
#endif
