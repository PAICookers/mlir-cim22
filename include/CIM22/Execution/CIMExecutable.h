//===- CIMTransaction.h - In-memory CIM22 execution plan --------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIM22_EXECUTION_CIMEXECUTABLE_H
#define CIM22_EXECUTION_CIMEXECUTABLE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace cim22::execution {

enum class CIMPacketKind {
  Control,
  Weight,
  Work,
  InputCacheWrite,
  ReturnRoute,
  OutputCacheRead,
};

struct CIMFramePacket {
  CIMPacketKind kind;
  std::array<int32_t, 6> route{};
  int64_t groupId = -1;
  int64_t workId = -1;
  int64_t macroSlot = -1;
  int64_t cacheAddress = -1;
  std::array<int32_t, 3> testCore{};
};

struct StaticWeightSection {
  int64_t groupId = -1;
  int64_t workId = -1;
  int64_t coreSlot = -1;
  int64_t macroSlot = -1;
  std::string resource;
  std::array<int32_t, 6> route{};
  std::vector<int32_t> words;
  std::array<int8_t, 16 * 64> values{};
};

struct DynamicInputBinding {
  int64_t groupId = -1;
  int64_t workId = -1;
  int64_t macroSlot = -1;
  int64_t inputSlot = -1;
};

struct ReadbackBinding {
  int64_t groupId = -1;
  int64_t workId = -1;
  int64_t macroSlot = -1;
  int64_t outputCacheAddress = -1;
  std::array<int32_t, 6> route{};
  std::array<int32_t, 3> testCore{};
};

struct CIMWork {
  int64_t workId = -1;
  int64_t coreSlot = -1;
  int64_t macroSlot = -1;
  std::array<int32_t, 6> route{};
};

struct CIMGroup {
  int64_t groupId = -1;
  std::vector<CIMWork> works;
};

class CIMTransaction final {
public:
  CIMTransaction(std::string targetProfile, int64_t targetProfileVersion,
                int64_t executionPlanVersion, std::vector<CIMGroup> groups,
                std::vector<StaticWeightSection> weights,
                std::vector<CIMFramePacket> packets,
                std::vector<DynamicInputBinding> inputs,
                std::vector<ReadbackBinding> readbacks,
                std::vector<uint64_t> flits)
      : targetProfile_(std::move(targetProfile)),
        targetProfileVersion_(targetProfileVersion),
        executionPlanVersion_(executionPlanVersion), groups_(std::move(groups)),
        weights_(std::move(weights)), packets_(std::move(packets)),
        inputs_(std::move(inputs)), readbacks_(std::move(readbacks)),
        flits_(std::move(flits)) {}

  llvm::StringRef getTargetProfile() const noexcept { return targetProfile_; }
  int64_t getTargetProfileVersion() const noexcept {
    return targetProfileVersion_;
  }
  int64_t getExecutionPlanVersion() const noexcept {
    return executionPlanVersion_;
  }
  llvm::ArrayRef<CIMGroup> getGroups() const { return groups_; }
  llvm::ArrayRef<StaticWeightSection> getStaticWeights() const {
    return weights_;
  }
  llvm::ArrayRef<CIMFramePacket> getPackets() const { return packets_; }
  llvm::ArrayRef<DynamicInputBinding> getDynamicInputs() const {
    return inputs_;
  }
  llvm::ArrayRef<ReadbackBinding> getReadbacks() const { return readbacks_; }
  llvm::ArrayRef<uint64_t> getFlits() const { return flits_; }

private:
  std::string targetProfile_;
  int64_t targetProfileVersion_ = 0;
  int64_t executionPlanVersion_ = 0;
  std::vector<CIMGroup> groups_;
  std::vector<StaticWeightSection> weights_;
  std::vector<CIMFramePacket> packets_;
  std::vector<DynamicInputBinding> inputs_;
  std::vector<ReadbackBinding> readbacks_;
  std::vector<uint64_t> flits_;
};

} // namespace cim22::execution

#endif // CIM22_EXECUTION_CIMEXECUTABLE_H
