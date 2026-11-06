//===- CIMSoftwareRunner.cpp - CIM22 software runner ------------*- C++ -*-===//

#include "CIM22/Execution/CIMRunner.h"

#include "CIM22/Support/BF16Support.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>

namespace cim22::execution {
namespace {
constexpr int64_t kCIMCoreCount = 20;
constexpr int64_t kCIMMacroCount = 2;
constexpr int64_t kCIMCacheRows = 8;
constexpr int64_t kCIMWeightElements = 16 * 64;

llvm::Error runnerError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

struct MacroState {
  std::array<int8_t, kCIMInputElements> input{};
  BF16InputPreparation bf16;
  // Store the final lane payload: signed INT21 or zero-extended BF16 bits.
  std::array<std::array<int32_t, kCIMOutputElements>, kCIMCacheRows>
      outputCache{};
};

size_t macroIndex(int64_t core, int64_t macro) {
  return static_cast<size_t>(core * kCIMMacroCount + macro);
}

const CIMWork *findWork(const CIMTransaction &executable, int64_t groupId,
                        int64_t workId) {
  for (const CIMGroup &group : executable.getGroups()) {
    if (group.groupId != groupId)
      continue;
    for (const CIMWork &work : group.works)
      if (work.workId == workId)
        return &work;
  }
  return nullptr;
}

const StaticWeightSection *findWeight(const CIMTransaction &executable,
                                      int64_t groupId, int64_t workId) {
  for (const StaticWeightSection &weight : executable.getStaticWeights())
    if (weight.groupId == groupId && weight.workId == workId)
      return &weight;
  return nullptr;
}

const DynamicInputBinding *findInput(const CIMTransaction &executable,
                                     int64_t groupId, int64_t workId) {
  for (const DynamicInputBinding &input : executable.getDynamicInputs())
    if (input.groupId == groupId && input.workId == workId)
      return &input;
  return nullptr;
}

const ReadbackBinding *findReadback(const CIMTransaction &executable,
                                    int64_t groupId, int64_t workId,
                                    int64_t macroSlot) {
  for (const ReadbackBinding &readback : executable.getReadbacks())
    if (readback.groupId == groupId && readback.workId == workId &&
        readback.macroSlot == macroSlot)
      return &readback;
  return nullptr;
}

bool hasReadbackSequence(const CIMTransaction &executable,
                         const ReadbackBinding &binding, size_t &packetIndex) {
  llvm::ArrayRef<CIMFramePacket> packets = executable.getPackets();
  for (size_t index = 0; index + 2 < packets.size(); ++index) {
    const CIMFramePacket &route = packets[index];
    const CIMFramePacket &control = packets[index + 1];
    const CIMFramePacket &read = packets[index + 2];
    if (route.kind != CIMPacketKind::ReturnRoute ||
        control.kind != CIMPacketKind::Control ||
        read.kind != CIMPacketKind::OutputCacheRead)
      continue;
    if (route.groupId != binding.groupId || route.workId != binding.workId ||
        route.macroSlot != binding.macroSlot || route.route != binding.route ||
        route.testCore != binding.testCore ||
        route.dataType != binding.dataType ||
        control.groupId != binding.groupId ||
        control.workId != binding.workId ||
        control.macroSlot != binding.macroSlot ||
        control.route != binding.route ||
        control.dataType != binding.dataType ||
        read.groupId != binding.groupId || read.workId != binding.workId ||
        read.macroSlot != binding.macroSlot || read.route != binding.route ||
        read.dataType != binding.dataType ||
        read.cacheAddress != binding.outputCacheAddress)
      continue;
    packetIndex = index;
    return true;
  }
  return false;
}

bool hasInputPacket(const CIMTransaction &executable,
                    const DynamicInputBinding &binding) {
  for (const CIMFramePacket &packet : executable.getPackets())
    if (packet.kind == CIMPacketKind::InputCacheWrite &&
        packet.groupId == binding.groupId && packet.workId == binding.workId &&
        packet.macroSlot == binding.macroSlot &&
        packet.dataType == binding.dataType)
      return true;
  return false;
}

bool hasWeightPacket(const CIMTransaction &executable,
                     const StaticWeightSection &weight,
                     CIMPacketKind kind = CIMPacketKind::Weight) {
  for (const CIMFramePacket &packet : executable.getPackets())
    if (packet.kind == kind && packet.groupId == weight.groupId &&
        packet.workId == weight.workId &&
        packet.macroSlot == weight.macroSlot &&
        packet.dataType == weight.dataType)
      return true;
  return false;
}

const char *validateWork(const CIMTransaction &transaction, const CIMWork &work,
                         const DynamicInputBinding *input,
                         const StaticWeightSection *weight,
                         const ReadbackBinding *readback) {
  if (!input || !weight)
    return "CIM work has incomplete input or weight binding";
  if (work.coreSlot < 0 || work.coreSlot >= kCIMCoreCount ||
      work.macroSlot < 0 || work.macroSlot >= kCIMMacroCount ||
      input->macroSlot != work.macroSlot || weight->coreSlot != work.coreSlot ||
      weight->macroSlot != work.macroSlot)
    return "CIM work has inconsistent Core or Macro placement";
  if (!hasInputPacket(transaction, *input))
    return "CIM input binding has no input Cache packet";
  if (!hasWeightPacket(transaction, *weight))
    return "CIM weight section has no weight packet";
  if (weight->dataType == CIMDataType::BF16 &&
      !hasWeightPacket(transaction, *weight, CIMPacketKind::WeightExponent))
    return "CIM BF16 weight section has no Weight_EXP packet";
  if (!readback)
    return "CIM work has no readback binding";
  if (input->dataType != weight->dataType ||
      readback->dataType != input->dataType)
    return "CIM work has inconsistent data mode";
  return nullptr;
}

bool hasReadbackPlacement(const ReadbackBinding &binding, const CIMWork *work) {
  return work && work->coreSlot >= 0 && work->coreSlot < kCIMCoreCount &&
         work->macroSlot == binding.macroSlot;
}

void emitTrace(CIMTraceSink *trace, uint64_t &sequence, CIMTraceEventKind kind,
               int64_t groupId = -1, int64_t workId = -1,
               int64_t macroSlot = -1, int64_t packetIndex = -1,
               int64_t cacheRowOrAddress = -1) {
  if (!trace)
    return;
  trace->record({sequence++, kind, groupId, workId, macroSlot, packetIndex,
                 cacheRowOrAddress});
}

llvm::Error validateBindings(const CIMTransaction &executable,
                             const CIMRunInputs &inputs,
                             CIMRunOutputs &outputs) {
  if (executable.getTargetProfile() != "cim22-4x5-v1" ||
      executable.getTargetProfileVersion() != 1 ||
      executable.getExecutionPlanVersion() != 1)
    return runnerError("unsupported CIM executable profile or version");
  if (inputs.values.size() != executable.getDynamicInputs().size())
    return runnerError("CIM input count does not match executable bindings");
  if (outputs.values.size() != executable.getReadbacks().size())
    return runnerError("CIM output count does not match executable bindings");
  for (auto [index, input] : llvm::enumerate(inputs.values)) {
    const DynamicInputBinding &binding = executable.getDynamicInputs()[index];
    if (binding.dataType == CIMDataType::Int8 &&
        input.values.size() != kCIMInputElements)
      return runnerError("CIM input view must contain exactly 64 INT8 values");
    if (binding.dataType == CIMDataType::BF16 &&
        input.bf16Values.size() != kCIMInputElements)
      return runnerError("CIM input view must contain exactly 64 BF16 values");
  }
  for (auto [index, output] : llvm::enumerate(outputs.values)) {
    const ReadbackBinding &binding = executable.getReadbacks()[index];
    if (binding.dataType == CIMDataType::Int8 &&
        output.values.size() != kCIMOutputElements)
      return runnerError(
          "CIM output view must contain exactly 16 INT32 values");
    if (binding.dataType == CIMDataType::BF16 &&
        output.bf16Values.size() != kCIMOutputElements)
      return runnerError("CIM output view must contain exactly 16 BF16 values");
  }
  for (const ReadbackBinding &binding : executable.getReadbacks())
    if (binding.macroSlot < 0 || binding.macroSlot >= kCIMMacroCount ||
        binding.outputCacheAddress < 0 ||
        binding.outputCacheAddress >= kCIMCacheRows)
      return runnerError("CIM readback binding is outside the target profile");
  return llvm::Error::success();
}
} // namespace

CIMSoftwareRunner::Plan::Plan(const CIMTransaction &transaction,
                              bool batchReadbacks)
    : transaction(transaction) {
  groups.reserve(transaction.getGroups().size());
  for (const CIMGroup &group : transaction.getGroups())
    groups.push_back({&group, {}});
  llvm::sort(groups, [](const Group &lhs, const Group &rhs) {
    return lhs.group->groupId < rhs.group->groupId;
  });

  // Readback may reorder by Macro within a wave; binding indices still
  // determine where each output lands.
  uint32_t waveCores = 0;
  bool waveBatchable = false;
  auto finishWave = [&] {
    if (!waveBatchable)
      return;
    auto &readbacks = waves.back().readbacks;
    std::stable_partition(
        readbacks.begin(), readbacks.end(), [&](const Readback &readback) {
          return transaction.getReadbacks()[readback.bindingIndex].macroSlot ==
                 0;
        });
  };

  // A later group on the same Core may overwrite Cache rows or change the
  // selected Macro, so finish its predecessor's readbacks first.
  for (auto [groupIndex, entry] : llvm::enumerate(groups)) {
    const CIMGroup &group = *entry.group;
    uint32_t groupCores = 0;
    bool groupBatchable = true;
    entry.works.reserve(group.works.size());
    for (const CIMWork &work : group.works) {
      const auto *input = findInput(transaction, group.groupId, work.workId);
      const auto *weight = findWeight(transaction, group.groupId, work.workId);
      const auto *readback =
          findReadback(transaction, group.groupId, work.workId, work.macroSlot);
      const char *error =
          validateWork(transaction, work, input, weight, readback);
      entry.works.push_back({input, weight, readback, error});
      if (error)
        groupBatchable = false;
      else
        groupCores |= uint32_t{1} << work.coreSlot;
    }

    llvm::SmallVector<Readback, 2> groupReadbacks;
    for (auto [index, binding] : llvm::enumerate(transaction.getReadbacks())) {
      if (binding.groupId != group.groupId)
        continue;
      size_t packetIndex = 0;
      const bool found = hasReadbackSequence(transaction, binding, packetIndex);
      const auto *work = findWork(transaction, binding.groupId, binding.workId);
      groupReadbacks.push_back(
          {index, work, found ? std::optional(packetIndex) : std::nullopt});
      if (!found || !hasReadbackPlacement(binding, work))
        groupBatchable = false;
      else
        groupCores |= uint32_t{1} << work->coreSlot;
    }

    // Keep malformed groups standalone, with their original error order.
    if (waves.empty() || !waveBatchable || !groupBatchable ||
        (waveCores & groupCores) != 0) {
      finishWave();
      waves.push_back({groupIndex + 1, {}});
      waveCores = 0;
      waveBatchable = batchReadbacks && groupBatchable;
    }
    Wave &wave = waves.back();
    wave.groupEnd = groupIndex + 1;
    wave.readbacks.insert(wave.readbacks.end(), groupReadbacks.begin(),
                          groupReadbacks.end());
    waveCores |= groupCores;
  }
  finishWave();
}

llvm::Error CIMSoftwareRunner::run(const CIMTransaction &executable,
                                   const CIMRunInputs &inputs,
                                   CIMRunOutputs &outputs,
                                   CIMTraceSink *trace) {
  return run(Plan(executable), inputs, outputs, trace);
}

llvm::Error CIMSoftwareRunner::run(const Plan &plan, const CIMRunInputs &inputs,
                                   CIMRunOutputs &outputs,
                                   CIMTraceSink *trace) {
  return runImpl(plan, inputs, outputs, nullptr, {}, trace);
}

llvm::Error CIMSoftwareRunner::run(const Plan &plan, const CIMRunInputs &inputs,
                                   CIMRunOutputs &outputs,
                                   CIMWeightResidency &residency,
                                   llvm::ArrayRef<CIMWeightIdentity> identities,
                                   CIMTraceSink *trace) {
  if (auto error =
          runImpl(plan, inputs, outputs, &residency, identities, trace)) {
    residency.clear();
    return error;
  }
  return llvm::Error::success();
}

llvm::Error CIMSoftwareRunner::runImpl(
    const Plan &plan, const CIMRunInputs &inputs, CIMRunOutputs &outputs,
    CIMWeightResidency *residency, llvm::ArrayRef<CIMWeightIdentity> identities,
    CIMTraceSink *trace) {
  const CIMTransaction &executable = plan.transaction;
  uint64_t sequence = 0;
  auto fail = [&](const llvm::Twine &message) -> llvm::Error {
    emitTrace(trace, sequence, CIMTraceEventKind::Error);
    return runnerError(message);
  };
  if (residency && identities.size() != executable.getStaticWeights().size())
    return fail("prepared CIM weight count differs from transaction");
  if (llvm::Error error = validateBindings(executable, inputs, outputs)) {
    emitTrace(trace, sequence, CIMTraceEventKind::Error);
    return error;
  }

  // Keep only configured payloads: an array of optional<MacroState> can still
  // compile to zeroing every slot's storage. Indices survive SmallVector growth
  // and retain each slot's Cache across groups, without retaining it across
  // runs.
  std::array<int, kCIMCoreCount * kCIMMacroCount> macroIndices;
  macroIndices.fill(-1);
  llvm::SmallVector<MacroState, 2> macros;
  emitTrace(trace, sequence, CIMTraceEventKind::BeginRun);

  auto executeGroup = [&](const Plan::Group &entry) -> llvm::Error {
    const CIMGroup &group = *entry.group;
    for (auto [work, resolved] : llvm::zip_equal(group.works, entry.works)) {
      if (resolved.error)
        return fail(resolved.error);
      const auto *input = resolved.input;
      const auto *weight = resolved.weight;
      const ReadbackBinding *readback = resolved.readback;
      const size_t physicalSlot = macroIndex(work.coreSlot, work.macroSlot);
      int &slot = macroIndices[physicalSlot];
      if (slot < 0) {
        slot = static_cast<int>(macros.size());
        macros.emplace_back();
      }
      MacroState &macro = macros[slot];
      const size_t inputPosition = input - executable.getDynamicInputs().data();
      if (input->dataType == CIMDataType::BF16) {
        auto prepared =
            prepareBF16Input(inputs.values[inputPosition].bf16Values, bf16Mode);
        if (!prepared)
          return fail(llvm::toString(prepared.takeError()));
        if (auto error = validateBF16Group(*weight, *prepared, bf16Mode))
          return fail(llvm::toString(std::move(error)));
        macro.bf16 = std::move(*prepared);
        macro.input = macro.bf16.aligned.mantissas;
      } else {
        std::copy(inputs.values[inputPosition].values.begin(),
                  inputs.values[inputPosition].values.end(),
                  macro.input.begin());
      }
      std::array<int8_t, kCIMWeightElements> coldWeight;
      std::array<uint8_t, kCIMOutputElements> coldExponents;
      const CIMWeightImage *resident = nullptr;
      bool wroteWeight = true;
      if (residency) {
        const auto &identity =
            identities[weight - executable.getStaticWeights().data()];
        if (!identity)
          return fail("missing prepared CIM weight identity");
        resident = residency->match(physicalSlot, identity, bf16Mode);
        wroteWeight = !resident;
        if (wroteWeight) {
          residency->store(physicalSlot, identity, bf16Mode);
          resident = identity.get();
          ++residency->stats.writes;
        } else {
          ++residency->stats.hits;
        }
      } else {
        coldWeight = weight->values;
        coldExponents = weight->exponents;
      }
      const auto &weightValues = residency ? resident->values : coldWeight;
      const auto &weightExponents =
          residency ? resident->exponents : coldExponents;
      emitTrace(trace, sequence, CIMTraceEventKind::ConfigureInput,
                group.groupId, work.workId, work.macroSlot);
      if (wroteWeight)
        emitTrace(trace, sequence, CIMTraceEventKind::ConfigureWeight,
                  group.groupId, work.workId, work.macroSlot);
      emitTrace(trace, sequence, CIMTraceEventKind::SelectMacro, group.groupId,
                work.workId, work.macroSlot);
      emitTrace(trace, sequence, CIMTraceEventKind::StartWork, group.groupId,
                work.workId, work.macroSlot);
      std::array<int32_t, kCIMOutputElements> result{};
      for (size_t row = 0; row < kCIMOutputElements; ++row) {
        auto laneWeight =
            llvm::ArrayRef(weightValues)
                .slice(row * kCIMInputElements, kCIMInputElements);
        const bool hasSupport = input->dataType != CIMDataType::BF16 ||
                                bf16Mode != BF16Compensation::SignSplit ||
                                hasBF16Support(laneWeight, macro.bf16.support);
        if (!hasSupport)
          continue;
        int32_t sum = 0;
        for (size_t k = 0; k < kCIMInputElements; ++k)
          sum += static_cast<int32_t>(laneWeight[k]) *
                 static_cast<int32_t>(macro.input[k]);
        if (sum < kCIMI21Min || sum >= kCIMI21MaxExclusive)
          return fail("CIM software result exceeds signed i21 range");
        if (input->dataType == CIMDataType::BF16) {
          const uint8_t chipExponent =
              bf16Mode == BF16Compensation::None ? weightExponents[row] : 0;
          const uint16_t chipBits = mlir::cim22::int21ToBF16(
              sum, chipExponent, macro.bf16.aligned.exponent);
          auto restored = restoreBF16Output(chipBits, weightExponents[row],
                                            hasSupport, macro.bf16, bf16Mode);
          if (!restored)
            return fail(llvm::toString(restored.takeError()));
          result[row] = *restored;
        } else {
          result[row] = sum;
        }
      }
      macro.outputCache[readback->outputCacheAddress] = result;
    }
    return llvm::Error::success();
  };

  size_t groupIndex = 0;
  for (const Plan::Wave &wave : plan.waves) {
    for (; groupIndex < wave.groupEnd; ++groupIndex)
      if (llvm::Error error = executeGroup(plan.groups[groupIndex]))
        return error;

    for (const Plan::Readback &resolved : wave.readbacks) {
      const size_t index = resolved.bindingIndex;
      const ReadbackBinding &binding = executable.getReadbacks()[index];
      if (!resolved.packetIndex)
        return fail("CIM readback is missing 1010/control/0101 sequence");
      const size_t packetIndex = *resolved.packetIndex;
      const CIMWork *work = resolved.work;
      if (!hasReadbackPlacement(binding, work))
        return fail("CIM readback references unknown placement");
      const int slot =
          macroIndices[macroIndex(work->coreSlot, work->macroSlot)];
      if (slot < 0)
        return fail("CIM readback references an unconfigured Macro");
      const MacroState &macro = macros[slot];
      emitTrace(trace, sequence, CIMTraceEventKind::ConfigureReturnRoute,
                binding.groupId, binding.workId, binding.macroSlot,
                static_cast<int64_t>(packetIndex));
      emitTrace(trace, sequence, CIMTraceEventKind::ReadOutput, binding.groupId,
                binding.workId, binding.macroSlot,
                static_cast<int64_t>(packetIndex + 2),
                binding.outputCacheAddress);
      if (binding.dataType == CIMDataType::BF16) {
        llvm::transform(
            macro.outputCache[binding.outputCacheAddress],
            outputs.values[index].bf16Values.begin(),
            [](int32_t value) { return static_cast<uint16_t>(value); });
      } else {
        std::copy(macro.outputCache[binding.outputCacheAddress].begin(),
                  macro.outputCache[binding.outputCacheAddress].end(),
                  outputs.values[index].values.begin());
      }
    }
  }

  emitTrace(trace, sequence, CIMTraceEventKind::Complete);
  return llvm::Error::success();
}

} // namespace cim22::execution
