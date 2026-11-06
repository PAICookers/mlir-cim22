// Internal state shared by the compilation adapter and execution-only runtime.
#ifndef CIM22_RUNTIME_SESSION_H
#define CIM22_RUNTIME_SESSION_H
#include "CIM22/Execution/CIMRunner.h"
#include "CIM22/Runtime/CIMBoardRunner.h"
#include "CIM22/Runtime/CIMDevice.h"
#include "ModelData.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include <memory>
#include <mutex>

namespace cim22::runtime {
// One loaded model, with per-backend state owned for its execution lifetime.
// Invoke, USB configuration and capture serialize through mutex.
struct Session {
  // Keeps either a JIT engine or an RTLD_LOCAL model DSO alive through invoke.
  std::shared_ptr<void> code;
  ModelData model;
  // Lazily built under the invoke lock. Plans borrow immutable transactions
  // from model and are destroyed before it.
  std::vector<std::optional<::cim22::execution::CIMSoftwareRunner::Plan>>
      softwarePlans;
  void (*invoke)(void **) = nullptr;
  std::unique_ptr<::cim22::runtime::CIMDevice> device;
  // The board runner borrows the device; destroy it first.
  std::unique_ptr<::cim22::runtime::CIMBoardRunner> runner;
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
_mlir_ciface_cim22_dispatch_sign_split(int64_t id,
                                       StridedMemRefType<uint16_t, 2> *input,
                                       StridedMemRefType<uint16_t, 2> *output);
extern "C" void
_mlir_ciface_cim22_dispatch_legacy(int64_t id,
                                   StridedMemRefType<uint16_t, 2> *input,
                                   StridedMemRefType<uint16_t, 2> *output);
extern "C" void
_mlir_ciface_cim22_dispatch_int8(int64_t id,
                                 StridedMemRefType<int8_t, 2> *input,
                                 StridedMemRefType<int32_t, 2> *output);
#endif
