//===- CIMRunner.h - CIM22 execution runners ------------------*- C++ -*-===//

#ifndef CIM22_EXECUTION_CIMRUNNER_H
#define CIM22_EXECUTION_CIMRUNNER_H

#include "CIM22/Execution/CIMExecutable.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>

namespace cim22::execution {

inline constexpr size_t kCIMInputElements = 64;
inline constexpr size_t kCIMOutputElements = 16;
inline constexpr int32_t kCIMI21Min = -(1 << 20);
inline constexpr int32_t kCIMI21MaxExclusive = 1 << 20;

struct CIMInputView {
  llvm::ArrayRef<int8_t> values;
};

struct CIMOutputView {
  llvm::MutableArrayRef<int32_t> values;
};

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

  virtual llvm::Error run(const CIMExecutable &executable,
                          const CIMRunInputs &inputs,
                          CIMRunOutputs &outputs,
                          CIMTraceSink *trace = nullptr) = 0;
};

class CIMSoftwareRunner final : public CIMRunner {
public:
  llvm::Error run(const CIMExecutable &executable, const CIMRunInputs &inputs,
                  CIMRunOutputs &outputs,
                  CIMTraceSink *trace = nullptr) override;
};

class CIMUartRunner final : public CIMRunner {
public:
  llvm::Error run(const CIMExecutable &executable, const CIMRunInputs &inputs,
                  CIMRunOutputs &outputs,
                  CIMTraceSink *trace = nullptr) override;
};

} // namespace cim22::execution

#endif // CIM22_EXECUTION_CIMRUNNER_H
