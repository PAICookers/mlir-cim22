// Internal state shared by the compilation adapter and execution-only runtime.
#ifndef CIM22_RUNTIME_SESSION_H
#define CIM22_RUNTIME_SESSION_H
#include "CIM22/Execution/CIMRunner.h"
#include "CIM22/Execution/WeightResidency.h"
#include "CIM22/Runtime/CIMBoardRunner.h"
#include "CIM22/Runtime/CIMDevice.h"
#include "CIM22/Runtime/ModelArtifact.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include <memory>
#include <mutex>

namespace cim22::runtime {
// One loaded model, with per-backend state owned for its execution lifetime.
// Invoke, USB configuration and capture serialize through mutex.
struct Session {
  ~Session() {
    // Release model code before compiler-owned temporary files are removed.
    runner.reset();
    device.reset();
    code.reset();
  }

  // Keeps the loaded model DSO alive through invoke.
  std::shared_ptr<void> code;
  ModelArtifact model;
  // Lazily built under the invoke lock. Plans borrow immutable transactions
  // from model and are destroyed before it.
  std::vector<std::optional<::cim22::execution::CIMSoftwareRunner::Plan>>
      softwarePlans;
  // Intern immutable images across transactions. Each prepared row follows
  // the corresponding transaction's StaticWeightSection order.
  ::cim22::execution::CIMWeightCatalog weightCatalog;
  std::vector<std::vector<::cim22::execution::CIMWeightIdentity>> weightPlans;
  // Simulator slots survive successful invocations. The USB device owns
  // its own physical residency, which ends when that device closes.
  ::cim22::execution::CIMWeightResidency softwareWeights;
  bool enableWeightResidency = true;
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
