// Internal state shared by the compilation adapter and execution-only runtime.
#ifndef CIM22_RUNTIME_SESSION_H
#define CIM22_RUNTIME_SESSION_H
#include "CIM22/Execution/CIMRunner.h"
#include "CIM22/Runtime/CIMDevice.h"
#include "ModelData.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include <memory>
#include <mutex>

namespace cim22::runtime {
struct Session {
  // Keeps either a JIT engine or an RTLD_LOCAL model DSO alive through invoke.
  std::shared_ptr<void> code;
  ModelData model;
  void (*invoke)(void **) = nullptr;
  std::unique_ptr<::cim22::runtime::CIMDevice> device;
  // Runner borrows the device; destroy it first.
  std::unique_ptr<::cim22::execution::CIMRunner> runner;
  std::string error;
  uint64_t dispatches = 0;
  std::mutex mutex;
};
void copyError(llvm::StringRef message, char *buffer, size_t capacity);
} // namespace cim22::runtime

extern "C" void
_mlir_ciface_cim22_dispatch(int64_t id, StridedMemRefType<uint16_t, 2> *input,
                            StridedMemRefType<uint16_t, 2> *output);
extern "C" void
_mlir_ciface_cim22_dispatch_int8(int64_t id,
                                 StridedMemRefType<int8_t, 2> *input,
                                 StridedMemRefType<int32_t, 2> *output);
#endif
