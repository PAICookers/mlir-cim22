//===- BF16Compensation.cpp - Shared CIM BF16 peripherals ------*- C++ -*-===//
#include "CIM22/Execution/BF16Compensation.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/bit.h"
#include <algorithm>
#include <cmath>

namespace cim22::execution {
namespace {
llvm::Error runnerError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

uint16_t scaleBF16Input(uint16_t bits, int exponent) {
  llvm::APFloat value(llvm::APFloat::BFloat(), llvm::APInt(16, bits));
  value = llvm::scalbn(value, exponent, llvm::APFloat::rmNearestTiesToEven);
  return static_cast<uint16_t>(value.bitcastToAPInt().getZExtValue());
}

// The effective biased exponent extends below zero for a subnormal maximum:
// frac * 2^-133 has floor(log2(value)) + 127 = highest_bit(frac) - 6.
int effectiveBF16Exponent(uint16_t maximum) {
  const int exponent = (maximum >> 7) & 0xff;
  if (exponent)
    return exponent;
  int highestBit = 0;
  for (uint16_t fraction = maximum & 0x7f; fraction > 1; fraction >>= 1)
    ++highestBit;
  return highestBit - 6;
}

llvm::Expected<uint16_t> compensateBF16Output(uint16_t bits, int exponent) {
  llvm::APFloat value(llvm::APFloat::BFloat(), llvm::APInt(16, bits));
  if (!value.isFinite())
    return runnerError("CIM BF16 chip output is not finite");
  bool losesInfo = false;
  value.convert(llvm::APFloat::IEEEdouble(), llvm::APFloat::rmNearestTiesToEven,
                &losesInfo);
  const double compensated = std::ldexp(value.convertToDouble(), exponent);
  if (!std::isfinite(compensated))
    return runnerError("CIM BF16 compensation is not finite");
  value = llvm::APFloat(compensated);
  // The existing tile ABI is BF16. Round once after binary64 power-of-two
  // compensation, before the generated Host IR widens/merges partial sums.
  value.convert(llvm::APFloat::BFloat(), llvm::APFloat::rmNearestTiesToEven,
                &losesInfo);
  if (!value.isFinite())
    return runnerError("CIM BF16 compensated tile exceeds BF16");
  return static_cast<uint16_t>(value.bitcastToAPInt().getZExtValue());
}

} // namespace

llvm::Error validateBF16Group(const StaticWeightSection &weight,
                              const BF16InputPreparation &input,
                              BF16Compensation mode) {
  if (mode == BF16Compensation::Legacy &&
      llvm::all_of(input.aligned.mantissas,
                   [](int8_t value) { return value == -128; })) {
    // A 64-term signed-i8 dot can exceed i21 only at +2^20: every pair
    // must be (-128,-128). Reject that pattern without computing the dot.
    for (size_t lane = 0; lane < weight.exponents.size(); ++lane)
      if (llvm::all_of(llvm::ArrayRef(weight.values).slice(lane * 64, 64),
                       [](int8_t value) { return value == -128; }))
        return runnerError("CIM BF16 result exceeds signed i21 range");
  }
  if (mode != BF16Compensation::SignSplit)
    return llvm::Error::success();
  if (llvm::is_contained(weight.exponents, uint8_t{255}))
    return runnerError("CIM nonnegative BF16 weight exponent is not finite");
  uint8_t bits = 0;
  for (int8_t value : weight.values)
    bits |= static_cast<uint8_t>(value);
  if (bits & 0x80)
    return runnerError(
        "CIM nonnegative BF16 mode rejects negative prepared weights");
  // Nonnegative aligned operands are 0..127: p<=64*127^2<2^20 fits i21.
  // For p>0, converter Ew'=0/Ei'=127 keeps its output exponent in 121..140.
  return llvm::Error::success();
}

llvm::Expected<BF16InputPreparation>
prepareBF16Input(llvm::ArrayRef<uint16_t> input, BF16Compensation mode) {
  BF16InputPreparation result;
  if (input.size() != result.chipValues.size())
    return runnerError("CIM input view must contain exactly 64 BF16 values");
  std::copy(input.begin(), input.end(), result.chipValues.begin());
  if (llvm::any_of(
          input, [](uint16_t value) { return ((value >> 7) & 0xff) == 0xff; }))
    return runnerError("CIM BF16 input does not support NaN or infinity");
  if (mode == BF16Compensation::SignSplit) {
    uint16_t maximum = 0;
    for (uint16_t &value : result.chipValues) {
      if ((value & 0x8000) && (value & 0x7fff))
        return runnerError("CIM nonnegative BF16 mode rejects negative inputs");
      value &= 0x7fff;
      maximum = std::max(maximum, value);
    }
    if (!maximum)
      return result;
    // Ex=floor(log2(max(x)))+127; x'=x*2^(127-Ex) puts the group max in [1,2).
    // One scale per 64 inputs; any scaling underflow is below alignment's LSB.
    result.effectiveExponent = effectiveBF16Exponent(maximum);
    const int shift = 127 - result.effectiveExponent;
    if (shift)
      for (uint16_t &value : result.chipValues)
        value = scaleBF16Input(value, shift);
    result.aligned = mlir::cim22::prealignBF16Vector(result.chipValues);
    if (result.aligned.exponent != 127)
      return runnerError("CIM nonnegative BF16 normalization did not produce "
                         "input exponent 127");
    for (size_t k = 0; k < result.aligned.mantissas.size(); ++k) {
      if (result.aligned.mantissas[k] < 0)
        return runnerError("CIM nonnegative BF16 normalization changed sign");
      if (result.aligned.mantissas[k])
        result.support |= uint64_t{1} << k;
    }
  } else {
    result.aligned = mlir::cim22::prealignBF16Vector(result.chipValues);
    if (mode == BF16Compensation::Legacy) {
      result.scaleDelta =
          std::max(0, static_cast<int>(result.aligned.exponent) - 240);
      if (result.scaleDelta) {
        const auto original = result.aligned.mantissas;
        for (uint16_t &value : result.chipValues)
          value = scaleBF16Input(value, -result.scaleDelta);
        result.aligned = mlir::cim22::prealignBF16Vector(result.chipValues);
        if (result.aligned.exponent > 240 ||
            result.aligned.mantissas != original)
          return runnerError("CIM legacy BF16 group scaling changed prealigned "
                             "input mantissas");
      }
    }
  }
  return result;
}

// A nonnegative dot is zero iff the quantized supports do not intersect.
// int21ToBF16(0,0,127)==int21ToBF16(128,0,127); readback cannot prove zero.
bool hasBF16Support(llvm::ArrayRef<int8_t> weight, uint64_t support) {
  for (; support; support &= support - 1)
    if (weight[llvm::countr_zero(support)])
      return true;
  return false;
}

llvm::Expected<uint16_t>
restoreBF16Output(uint16_t bits, uint8_t weightExponent, bool hasSupport,
                  const BF16InputPreparation &input, BF16Compensation mode) {
  if (mode == BF16Compensation::None)
    return bits;
  if (mode == BF16Compensation::SignSplit && !hasSupport)
    return uint16_t{0};
  // SignSplit scales: Sx=2^(Ex-133), Sw=2^(Ew-133), where 133=127+6.
  // Chip c=Q(p)/64; Q retains p's highest eight significant integer bits.
  // Thus y=c*2^(Ew+Ex-260), with 260=2*133-6; Ew is raw, Ex is effective.
  // The BF16 tile rounding occurs before Host K reductions and branch merges.
  const int exponent =
      mode == BF16Compensation::SignSplit
          ? static_cast<int>(weightExponent) + input.effectiveExponent - 260
          : static_cast<int>(weightExponent) + input.scaleDelta - 133;
  return compensateBF16Output(bits, exponent);
}
} // namespace cim22::execution
