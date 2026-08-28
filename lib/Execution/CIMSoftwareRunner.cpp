//===- CIMSoftwareRunner.cpp - CIM22 software runner ------------*- C++ -*-===//

#include "CIM22/Execution/CIMRunner.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <numeric>

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
  std::array<int8_t, kCIMWeightElements> weight{};
  std::array<std::array<int32_t, kCIMOutputElements>, kCIMCacheRows>
      outputCache{};
};

size_t macroIndex(int64_t core, int64_t macro) {
  return static_cast<size_t>(core * kCIMMacroCount + macro);
}

const CIMWork *findWork(const CIMExecutable &executable, int64_t groupId,
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

const StaticWeightSection *findWeight(const CIMExecutable &executable,
                                      int64_t groupId, int64_t workId) {
  for (const StaticWeightSection &weight : executable.getStaticWeights())
    if (weight.groupId == groupId && weight.workId == workId)
      return &weight;
  return nullptr;
}

const DynamicInputBinding *findInput(const CIMExecutable &executable,
                                     int64_t groupId, int64_t workId) {
  for (const DynamicInputBinding &input : executable.getDynamicInputs())
    if (input.groupId == groupId && input.workId == workId)
      return &input;
  return nullptr;
}

const ReadbackBinding *findReadback(const CIMExecutable &executable,
                                    int64_t groupId, int64_t workId,
                                    int64_t macroSlot) {
  for (const ReadbackBinding &readback : executable.getReadbacks())
    if (readback.groupId == groupId && readback.workId == workId &&
        readback.macroSlot == macroSlot)
      return &readback;
  return nullptr;
}

bool hasReadbackSequence(const CIMExecutable &executable,
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
        control.groupId != binding.groupId ||
        control.workId != binding.workId ||
        control.macroSlot != binding.macroSlot ||
        control.route != binding.route || read.groupId != binding.groupId ||
        read.workId != binding.workId || read.macroSlot != binding.macroSlot ||
        read.route != binding.route ||
        read.cacheAddress != binding.outputCacheAddress)
      continue;
    packetIndex = index;
    return true;
  }
  return false;
}

bool hasInputPacket(const CIMExecutable &executable,
                    const DynamicInputBinding &binding) {
  for (const CIMFramePacket &packet : executable.getPackets())
    if (packet.kind == CIMPacketKind::InputCacheWrite &&
        packet.groupId == binding.groupId && packet.workId == binding.workId &&
        packet.macroSlot == binding.macroSlot)
      return true;
  return false;
}

bool hasWeightPacket(const CIMExecutable &executable,
                     const StaticWeightSection &weight) {
  for (const CIMFramePacket &packet : executable.getPackets())
    if (packet.kind == CIMPacketKind::Weight &&
        packet.groupId == weight.groupId && packet.workId == weight.workId &&
        packet.macroSlot == weight.macroSlot)
      return true;
  return false;
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

llvm::Error validateBindings(const CIMExecutable &executable,
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
  for (const CIMInputView &input : inputs.values)
    if (input.values.size() != kCIMInputElements)
      return runnerError("CIM input view must contain exactly 64 INT8 values");
  for (const CIMOutputView &output : outputs.values)
    if (output.values.size() != kCIMOutputElements)
      return runnerError(
          "CIM output view must contain exactly 16 INT32 values");
  for (const ReadbackBinding &binding : executable.getReadbacks())
    if (binding.macroSlot < 0 || binding.macroSlot >= kCIMMacroCount ||
        binding.outputCacheAddress < 0 ||
        binding.outputCacheAddress >= kCIMCacheRows)
      return runnerError("CIM readback binding is outside the INT8 profile");
  return llvm::Error::success();
}
} // namespace

llvm::Error CIMSoftwareRunner::run(const CIMExecutable &executable,
                                   const CIMRunInputs &inputs,
                                   CIMRunOutputs &outputs,
                                   CIMTraceSink *trace) {
  uint64_t sequence = 0;
  auto fail = [&](const llvm::Twine &message) -> llvm::Error {
    emitTrace(trace, sequence, CIMTraceEventKind::Error);
    return runnerError(message);
  };
  if (llvm::Error error = validateBindings(executable, inputs, outputs)) {
    emitTrace(trace, sequence, CIMTraceEventKind::Error);
    return error;
  }

  std::array<MacroState, kCIMCoreCount * kCIMMacroCount> macros{};
  emitTrace(trace, sequence, CIMTraceEventKind::BeginRun);

  std::vector<size_t> groupOrder(executable.getGroups().size());
  std::iota(groupOrder.begin(), groupOrder.end(), 0);
  llvm::sort(groupOrder, [&](size_t lhs, size_t rhs) {
    return executable.getGroups()[lhs].groupId <
           executable.getGroups()[rhs].groupId;
  });

  for (size_t groupIndex : groupOrder) {
    const CIMGroup &group = executable.getGroups()[groupIndex];
    for (const CIMWork &work : group.works) {
      const DynamicInputBinding *input =
          findInput(executable, group.groupId, work.workId);
      const StaticWeightSection *weight =
          findWeight(executable, group.groupId, work.workId);
      if (!input || !weight)
        return fail("CIM work has incomplete input or weight binding");
      if (work.coreSlot < 0 || work.coreSlot >= kCIMCoreCount ||
          work.macroSlot < 0 || work.macroSlot >= kCIMMacroCount ||
          input->macroSlot != work.macroSlot ||
          weight->coreSlot != work.coreSlot ||
          weight->macroSlot != work.macroSlot)
        return fail("CIM work has inconsistent Core or Macro placement");
      if (!hasInputPacket(executable, *input))
        return fail("CIM input binding has no input Cache packet");
      if (!hasWeightPacket(executable, *weight))
        return fail("CIM weight section has no weight packet");
      const ReadbackBinding *readback =
          findReadback(executable, group.groupId, work.workId, work.macroSlot);
      if (!readback)
        return fail("CIM work has no readback binding");
      MacroState &macro = macros[macroIndex(work.coreSlot, work.macroSlot)];
      size_t inputPosition = 0;
      bool foundInput = false;
      for (auto [index, candidate] :
           llvm::enumerate(executable.getDynamicInputs())) {
        if (&candidate == input) {
          inputPosition = index;
          foundInput = true;
          break;
        }
      }
      if (!foundInput)
        return fail("CIM work input binding is not addressable");
      std::copy(inputs.values[inputPosition].values.begin(),
                inputs.values[inputPosition].values.end(), macro.input.begin());
      macro.weight = weight->values;
      emitTrace(trace, sequence, CIMTraceEventKind::ConfigureInput,
                group.groupId, work.workId, work.macroSlot);
      emitTrace(trace, sequence, CIMTraceEventKind::ConfigureWeight,
                group.groupId, work.workId, work.macroSlot);
      emitTrace(trace, sequence, CIMTraceEventKind::SelectMacro, group.groupId,
                work.workId, work.macroSlot);
      emitTrace(trace, sequence, CIMTraceEventKind::StartWork, group.groupId,
                work.workId, work.macroSlot);
      std::array<int32_t, kCIMOutputElements> result{};
      for (size_t row = 0; row < kCIMOutputElements; ++row) {
        int32_t sum = 0;
        for (size_t k = 0; k < kCIMInputElements; ++k)
          sum +=
              static_cast<int32_t>(macro.weight[row * kCIMInputElements + k]) *
              static_cast<int32_t>(macro.input[k]);
        if (sum < kCIMI21Min || sum >= kCIMI21MaxExclusive)
          return fail("CIM software result exceeds signed i21 range");
        result[row] = sum;
      }
      macro.outputCache[readback->outputCacheAddress] = result;
    }

    for (auto [index, binding] : llvm::enumerate(executable.getReadbacks())) {
      if (binding.groupId != group.groupId)
        continue;
      size_t packetIndex = 0;
      if (!hasReadbackSequence(executable, binding, packetIndex))
        return fail("CIM readback is missing 1010/control/0101 sequence");
      const CIMWork *work =
          findWork(executable, binding.groupId, binding.workId);
      if (!work || work->coreSlot < 0 || work->coreSlot >= kCIMCoreCount ||
          work->macroSlot != binding.macroSlot)
        return fail("CIM readback references unknown placement");
      const MacroState &macro =
          macros[macroIndex(work->coreSlot, work->macroSlot)];
      emitTrace(trace, sequence, CIMTraceEventKind::ConfigureReturnRoute,
                binding.groupId, binding.workId, binding.macroSlot,
                static_cast<int64_t>(packetIndex));
      emitTrace(trace, sequence, CIMTraceEventKind::ReadOutput, binding.groupId,
                binding.workId, binding.macroSlot,
                static_cast<int64_t>(packetIndex + 2),
                binding.outputCacheAddress);
      std::copy(macro.outputCache[binding.outputCacheAddress].begin(),
                macro.outputCache[binding.outputCacheAddress].end(),
                outputs.values[index].values.begin());
    }
  }

  emitTrace(trace, sequence, CIMTraceEventKind::Complete);
  return llvm::Error::success();
}

} // namespace cim22::execution
