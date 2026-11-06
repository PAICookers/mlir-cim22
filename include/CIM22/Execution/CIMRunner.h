//===- CIMRunner.h - CIM22 execution runners ------------------*- C++ -*-===//

#ifndef CIM22_EXECUTION_CIMRUNNER_H
#define CIM22_EXECUTION_CIMRUNNER_H

#include "CIM22/Execution/BF16Compensation.h"
#include "CIM22/Execution/CIMExecutable.h"
#include "CIM22/Execution/WeightResidency.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace cim22::execution {

inline constexpr size_t kCIMInputElements = 64;
inline constexpr size_t kCIMOutputElements = 16;
inline constexpr int32_t kCIMI21Min = -(1 << 20);
inline constexpr int32_t kCIMI21MaxExclusive = 1 << 20;

// Borrowed row of 64 elements. The input binding selects INT8 or raw BF16.
struct CIMInputView {
  llvm::ArrayRef<int8_t> values;
  llvm::ArrayRef<uint16_t> bf16Values;
};

// Borrowed writable row of 16 elements. The readback binding selects INT32
// or raw BF16.
struct CIMOutputView {
  llvm::MutableArrayRef<int32_t> values;
  llvm::MutableArrayRef<uint16_t> bf16Values;
};

// Views are supplied in dynamic-input or readback binding order respectively.
struct CIMRunInputs {
  llvm::ArrayRef<CIMInputView> values;
};

struct CIMRunOutputs {
  llvm::MutableArrayRef<CIMOutputView> values;
};

enum class CIMTraceEventKind {
  BeginRun,
  ConfigureWeight,
  ConfigureInput,
  SelectMacro,
  StartWork,
  ConfigureReturnRoute,
  ReadOutput,
  Complete,
  Error,
};

// Sequence is local to a run; unused placement and packet fields are -1.
struct CIMTraceEvent {
  uint64_t sequence = 0;
  CIMTraceEventKind kind = CIMTraceEventKind::Error;
  int64_t groupId = -1;
  int64_t workId = -1;
  int64_t macroSlot = -1;
  int64_t packetIndex = -1;
  int64_t cacheRowOrAddress = -1;
};

class CIMTraceSink {
public:
  virtual ~CIMTraceSink() = default;
  virtual void record(const CIMTraceEvent &event) = 0;
};

class CIMRunner {
public:
  virtual ~CIMRunner() = default;

  // Execute a transaction synchronously. The caller owns the views and any
  // trace sink. Outputs may be partially written if execution fails.
  virtual llvm::Error run(const CIMTransaction &executable,
                          const CIMRunInputs &inputs, CIMRunOutputs &outputs,
                          CIMTraceSink *trace = nullptr) = 0;
};

class CIMSoftwareRunner final : public CIMRunner {
public:
  // Resolve static bindings and packet indices once. The immutable
  // transaction must stay at the same address and outlive Plan. Batched
  // readbacks require disjoint input/output views and disjoint output rows.
  class Plan {
  public:
    explicit Plan(const CIMTransaction &transaction,
                  bool batchReadbacks = false);
    Plan(const CIMTransaction &&, bool batchReadbacks = false) = delete;

  private:
    friend class CIMSoftwareRunner;

    struct Work {
      const DynamicInputBinding *input;
      const StaticWeightSection *weight;
      const ReadbackBinding *readback;
      const char *error;
    };

    struct Readback {
      size_t bindingIndex;
      const CIMWork *work;
      std::optional<size_t> packetIndex;
    };

    struct Group {
      const CIMGroup *group;
      std::vector<Work> works;
    };

    struct Wave {
      size_t groupEnd;
      std::vector<Readback> readbacks;
    };

    const CIMTransaction &transaction;
    std::vector<Group> groups;
    std::vector<Wave> waves;
  };

  // SignSplit requires nonnegative branches; Legacy relocates only scale.
  // None retains original bit-level results, including exceptional patterns.
  explicit CIMSoftwareRunner(BF16Compensation mode = BF16Compensation::None)
      : bf16Mode(mode) {}

  llvm::Error run(const CIMTransaction &executable, const CIMRunInputs &inputs,
                  CIMRunOutputs &outputs,
                  CIMTraceSink *trace = nullptr) override;
  // Reuse static resolution with fresh dynamic inputs and Cache state.
  llvm::Error run(const Plan &plan, const CIMRunInputs &inputs,
                  CIMRunOutputs &outputs, CIMTraceSink *trace = nullptr);
  // Identities follow the transaction's StaticWeightSection order and
  // represent those exact payloads. The caller retains residency across
  // runs; failure clears resident weights.
  llvm::Error run(const Plan &plan, const CIMRunInputs &inputs,
                  CIMRunOutputs &outputs, CIMWeightResidency &residency,
                  llvm::ArrayRef<CIMWeightIdentity> identities,
                  CIMTraceSink *trace = nullptr);

private:
  llvm::Error runImpl(const Plan &plan, const CIMRunInputs &inputs,
                      CIMRunOutputs &outputs, CIMWeightResidency *residency,
                      llvm::ArrayRef<CIMWeightIdentity> identities,
                      CIMTraceSink *trace);

  BF16Compensation bf16Mode;
};

} // namespace cim22::execution

#endif // CIM22_EXECUTION_CIMRUNNER_H
