// Internal state shared by the compilation adapter and execution-only runtime.
#ifndef CIM22_RUNTIME_SESSION_H
#define CIM22_RUNTIME_SESSION_H
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
  std::string error;
  uint64_t dispatches = 0;
  std::mutex mutex;
};
void copyError(llvm::StringRef message, char *buffer, size_t capacity);
} // namespace cim22::runtime

extern "C" void
_mlir_ciface_cim22_dispatch(int64_t id, StridedMemRefType<uint16_t, 2> *input,
                            StridedMemRefType<uint16_t, 2> *output);
#endif
