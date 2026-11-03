//===- BF16Compensation.h - Shared CIM BF16 peripherals --------*- C++ -*-===//
#ifndef CIM22_EXECUTION_BF16COMPENSATION_H
#define CIM22_EXECUTION_BF16COMPENSATION_H

#include "CIM22/Execution/CIMExecutable.h"
#include "CIM22/Support/BF16Support.h"
#include "llvm/Support/Error.h"

namespace cim22::execution {
enum class BF16Compensation { None, Legacy, SignSplit };

// Per-group state for one invocation. The transaction keeps original Ew even
// when zero is used for the chip's converter configuration.
struct BF16InputPreparation {
  std::array<uint16_t, mlir::cim22::kBF16InputElements> chipValues{};
  mlir::cim22::BF16PrealignedVector aligned;
  uint64_t support = 0;
  int effectiveExponent = 0;
  int scaleDelta = 0;
};

llvm::Error validateBF16Group(const StaticWeightSection &weight,
                              const BF16InputPreparation &input,
                              BF16Compensation mode);
llvm::Expected<BF16InputPreparation>
prepareBF16Input(llvm::ArrayRef<uint16_t> input, BF16Compensation mode);
bool hasBF16Support(llvm::ArrayRef<int8_t> laneWeight, uint64_t inputSupport);

// Consume returned BF16 bits, never an integer or Host dot product. None
// preserves bits; SignSplit uses the caller's quantized-support check for zero.
llvm::Expected<uint16_t>
restoreBF16Output(uint16_t bits, uint8_t weightExponent, bool hasSupport,
                  const BF16InputPreparation &input, BF16Compensation mode);
} // namespace cim22::execution
#endif
