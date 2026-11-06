//===- WeightResidency.h - Session-local CIM weight identity ----*- C++ -*-===//

#ifndef CIM22_EXECUTION_WEIGHTRESIDENCY_H
#define CIM22_EXECUTION_WEIGHTRESIDENCY_H

#include "CIM22/Execution/BF16Compensation.h"
#include "CIM22/Execution/CIMExecutable.h"

#include "llvm/ADT/Hashing.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

namespace cim22::execution {

// One immutable weight payload, including the route, device words and
// simulator values. The physical slot belongs to the residency table.
struct CIMWeightImage {
  std::vector<int32_t> words;
  std::array<int8_t, 16 * 64> values;
  std::array<uint8_t, 16> exponents;
  std::array<int32_t, 6> route;
  CIMDataType dataType;
};

using CIMWeightIdentity = std::shared_ptr<const CIMWeightImage>;

// Intern exact payloads within a session. Hashes only select a candidate
// bucket; equality compares every field that defines the image.
class CIMWeightCatalog {
public:
  CIMWeightIdentity intern(const StaticWeightSection &weight) {
    auto hashRange = [](const auto &range) {
      return llvm::hash_combine_range(range.begin(), range.end());
    };
    auto &bucket = images[llvm::hash_combine(
        hashRange(weight.words), hashRange(weight.values),
        hashRange(weight.exponents), hashRange(weight.route),
        static_cast<unsigned>(weight.dataType))];
    for (const auto &image : bucket)
      if (image->words == weight.words && image->values == weight.values &&
          image->exponents == weight.exponents &&
          image->route == weight.route && image->dataType == weight.dataType)
        return image;
    return bucket.emplace_back(std::make_shared<const CIMWeightImage>(
        CIMWeightImage{weight.words, weight.values, weight.exponents,
                       weight.route, weight.dataType}));
  }

private:
  std::unordered_map<size_t, std::vector<CIMWeightIdentity>> images;
};

// Counts backend weight writes and reuse hits, including tracked operations
// completed before a subsequent failure.
struct CIMWeightStats {
  uint64_t writes = 0;
  uint64_t hits = 0;
};

// Tracks one weight image and compensation mode per physical Core/Macro slot
// (core * 2 + macro). The owning session or device serializes access.
class CIMWeightResidency {
public:
  static constexpr size_t slotCount = 40;

  // Caller supplies a valid slot. A match returns a borrowed image until the
  // slot changes; a miss returns nullptr.
  const CIMWeightImage *match(size_t slot, const CIMWeightIdentity &image,
                              BF16Compensation mode) const {
    const Entry &resident = slots[slot];
    return resident.mode == mode && resident.image == image
               ? resident.image.get()
               : nullptr;
  }

  // Replace the image in a valid slot and retain shared ownership of it.
  void store(size_t slot, const CIMWeightIdentity &image,
             BF16Compensation mode) {
    slots[slot] = {image, mode};
  }

  // Drop all resident images and keep the operation counters.
  void clear() { slots.fill({}); }

  CIMWeightStats stats{};

private:
  struct Entry {
    CIMWeightIdentity image;
    BF16Compensation mode = BF16Compensation::None;
  };
  std::array<Entry, slotCount> slots{};
};

} // namespace cim22::execution

#endif // CIM22_EXECUTION_WEIGHTRESIDENCY_H
