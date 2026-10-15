//===- CIMBoardRunner.cpp - CIM22 hardware runner -------------*- C++ -*-===//

#include "CIM22/Runtime/CIMBoardRunner.h"

#include "CIM22/Runtime/CIMPacketCodec.h"
#include "CIM22/Support/BF16Support.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace cim22::runtime {
namespace {
using namespace cim22::execution;

llvm::Error runnerError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

bool hasBF16(const CIMTransaction &executable) {
  return llvm::any_of(executable.getDynamicInputs(),
                      [](const DynamicInputBinding &binding) {
                        return binding.dataType == CIMDataType::BF16;
                      });
}

const CIMWork *findWork(const CIMTransaction &executable, int64_t groupId,
                        int64_t workId) {
  for (const CIMGroup &group : executable.getGroups())
    if (group.groupId == groupId)
      for (const CIMWork &work : group.works)
        if (work.workId == workId)
          return &work;
  return nullptr;
}

const StaticWeightSection *findWeight(const CIMTransaction &executable,
                                      int64_t groupId, int64_t workId) {
  auto it = llvm::find_if(
      executable.getStaticWeights(), [&](const StaticWeightSection &weight) {
        return weight.groupId == groupId && weight.workId == workId;
      });
  return it == executable.getStaticWeights().end() ? nullptr : &*it;
}

const DynamicInputBinding *findInput(const CIMTransaction &executable,
                                     int64_t groupId, int64_t workId,
                                     size_t &index) {
  for (auto [candidateIndex, input] :
       llvm::enumerate(executable.getDynamicInputs())) {
    if (input.groupId == groupId && input.workId == workId) {
      index = candidateIndex;
      return &input;
    }
  }
  return nullptr;
}

const ReadbackBinding *findReadback(const CIMTransaction &executable,
                                    int64_t groupId, int64_t workId,
                                    size_t &index) {
  for (auto [candidateIndex, readback] :
       llvm::enumerate(executable.getReadbacks())) {
    if (readback.groupId == groupId && readback.workId == workId) {
      index = candidateIndex;
      return &readback;
    }
  }
  return nullptr;
}

struct WorkBindings {
  const DynamicInputBinding *input = nullptr;
  const StaticWeightSection *weight = nullptr;
  const ReadbackBinding *readback = nullptr;
  size_t inputIndex = 0;
};

llvm::Expected<WorkBindings>
resolveWorkBindings(const CIMTransaction &executable, const CIMGroup &group,
                    const CIMWork &work) {
  WorkBindings bindings;
  size_t readbackIndex = 0;
  bindings.input =
      findInput(executable, group.groupId, work.workId, bindings.inputIndex);
  bindings.weight = findWeight(executable, group.groupId, work.workId);
  bindings.readback =
      findReadback(executable, group.groupId, work.workId, readbackIndex);
  if (!bindings.input || !bindings.weight || !bindings.readback)
    return runnerError("CIM work has incomplete runtime bindings");
  if (work.coreSlot < 0 || work.coreSlot >= 20 || work.macroSlot < 0 ||
      work.macroSlot >= 2 || bindings.input->macroSlot != work.macroSlot ||
      bindings.weight->coreSlot != work.coreSlot ||
      bindings.weight->macroSlot != work.macroSlot ||
      bindings.readback->macroSlot != work.macroSlot)
    return runnerError("CIM work has inconsistent Core or Macro placement");
  if (bindings.input->dataType != bindings.weight->dataType ||
      bindings.input->dataType != bindings.readback->dataType)
    return runnerError("CIM work has inconsistent data mode");
  return bindings;
}

void emitTrace(CIMTraceSink *trace, uint64_t &sequence, CIMTraceEventKind kind,
               int64_t groupId = -1, int64_t workId = -1,
               int64_t macroSlot = -1, int64_t packetIndex = -1,
               int64_t cacheAddress = -1) {
  if (!trace)
    return;
  trace->record({sequence++, kind, groupId, workId, macroSlot, packetIndex,
                 cacheAddress});
}

llvm::Error validateViews(const CIMTransaction &executable,
                          const CIMRunInputs &inputs, CIMRunOutputs &outputs) {
  if (executable.getTargetProfile() != "cim22-4x5-v1" ||
      executable.getTargetProfileVersion() != 1 ||
      executable.getExecutionPlanVersion() != 1)
    return runnerError("unsupported CIM executable profile or version");
  if (inputs.values.size() != executable.getDynamicInputs().size())
    return runnerError("CIM input count does not match executable bindings");
  if (outputs.values.size() != executable.getReadbacks().size())
    return runnerError("CIM output count does not match executable bindings");

  for (const CIMGroup &group : executable.getGroups()) {
    if (group.works.empty())
      return runnerError("CIM group has no work");
    const CIMWork &first = group.works.front();
    if (first.coreSlot < 0 || first.coreSlot >= 20)
      return runnerError("CIM work Core is outside [0, 19]");
    if (llvm::Error error = validateCIMRoute(first.route))
      return std::move(error);
    for (const CIMWork &work : group.works) {
      if (work.coreSlot != first.coreSlot || work.route != first.route)
        return runnerError(
            "CIM group works must share one Core and Once route");
      if (work.macroSlot < 0 || work.macroSlot >= 2)
        return runnerError("CIM work Macro is outside [0, 1]");
    }
  }

  for (auto [index, binding] : llvm::enumerate(executable.getDynamicInputs())) {
    const CIMInputView &input = inputs.values[index];
    if (binding.inputCacheAddress < 0 || binding.inputCacheAddress >= 8)
      return runnerError("CIM input Cache binding is outside [0, 7]");
    if (binding.dataType == CIMDataType::Int8 &&
        input.values.size() != kCIMInputElements)
      return runnerError("CIM input view must contain exactly 64 INT8 values");
    if (binding.dataType == CIMDataType::BF16 &&
        input.bf16Values.size() != kCIMInputElements)
      return runnerError("CIM input view must contain exactly 64 BF16 values");
  }
  for (auto [index, binding] : llvm::enumerate(executable.getReadbacks())) {
    const CIMOutputView &output = outputs.values[index];
    if (binding.outputCacheAddress < 0 || binding.outputCacheAddress >= 8)
      return runnerError("CIM output Cache binding is outside [0, 7]");
    if (binding.dataType == CIMDataType::Int8 &&
        output.values.size() != kCIMOutputElements)
      return runnerError(
          "CIM output view must contain exactly 16 INT32 values");
    if (binding.dataType == CIMDataType::BF16 &&
        output.bf16Values.size() != kCIMOutputElements)
      return runnerError("CIM output view must contain exactly 16 BF16 values");
    if (llvm::Error error = validateCIMRoute(binding.route))
      return std::move(error);
    for (int32_t value : binding.testCore)
      if (value < -31 || value > 31)
        return runnerError("CIM readback test Core is outside [-31, 31]");
  }
  return llvm::Error::success();
}

llvm::Expected<CIMDeviceProgram>
materializeProgram(const CIMTransaction &executable, const CIMRunInputs &inputs,
                   const CIMBoardRunPolicy &policy, CIMTraceSink *trace,
                   uint64_t &sequence) {
  CIMDeviceProgram program;

  // The supplier board flow configures one return path per active Core before
  // selecting or programming any Macro. Multiple Macro works on the same Core
  // therefore share one return-route configuration.
  CIMDeviceOperation returnRoutes;
  std::array<const ReadbackBinding *, 20> configuredRoutes{};
  std::array<int64_t, 20> configuredGroups{};
  configuredGroups.fill(-1);
  for (const CIMGroup &group : executable.getGroups()) {
    for (const CIMWork &work : group.works) {
      auto bindings = resolveWorkBindings(executable, group, work);
      if (!bindings)
        return bindings.takeError();
      const ReadbackBinding *readback = bindings->readback;
      const ReadbackBinding *&configured = configuredRoutes[work.coreSlot];
      if (configured) {
        if (configuredGroups[work.coreSlot] != group.groupId)
          return runnerError("CIM program has multiple groups on one Core");
        if (configured->route != readback->route ||
            configured->testCore != readback->testCore)
          return runnerError(
              "CIM Core has conflicting return-route configurations");
        continue;
      }
      configured = readback;
      configuredGroups[work.coreSlot] = group.groupId;
      returnRoutes.flits.push_back(encodeCIMReturnRouteFlit(
          readback->route, work.coreSlot, readback->testCore));
      emitTrace(trace, sequence, CIMTraceEventKind::ConfigureReturnRoute,
                readback->groupId, readback->workId, readback->macroSlot);
    }
  }
  if (returnRoutes.flits.empty())
    return runnerError("CIM program has no active Core return route");
  program.operations.push_back(std::move(returnRoutes));

  for (const CIMGroup &group : executable.getGroups()) {
    for (const CIMWork &work : group.works) {
      auto bindings = resolveWorkBindings(executable, group, work);
      if (!bindings)
        return bindings.takeError();
      const DynamicInputBinding &input = *bindings->input;
      const StaticWeightSection &weight = *bindings->weight;

      CIMDeviceOperation select;
      select.flits.push_back(
          encodeCIMControlFlit(work.route, work.macroSlot, input.dataType));
      if (input.dataType == CIMDataType::BF16) {
        const auto exponentWords =
            mlir::cim22::packBF16WeightExponents(weight.exponents);
        if (llvm::Error error = appendCIMWeightExponentPacket(
                weight.route, exponentWords, select.flits))
          return std::move(error);
      }
      program.operations.push_back(std::move(select));
      emitTrace(trace, sequence, CIMTraceEventKind::SelectMacro, group.groupId,
                work.workId, work.macroSlot);

      std::array<uint64_t, 16> inputWords{};
      if (input.dataType == CIMDataType::BF16) {
        std::array<uint16_t, kCIMInputElements> values{};
        std::copy(inputs.values[bindings->inputIndex].bf16Values.begin(),
                  inputs.values[bindings->inputIndex].bf16Values.end(),
                  values.begin());
        inputWords = mlir::cim22::packBF16InputCacheRow(values);
      } else {
        inputWords = packCIMInt8InputCacheRow(
            inputs.values[bindings->inputIndex].values);
      }
      CIMDeviceOperation inputWrite;
      inputWrite.postDelayMs = policy.cacheWriteDelayMs;
      if (llvm::Error error =
              appendCIMInputCachePacket(work.route, input.inputCacheAddress,
                                        inputWords, inputWrite.flits))
        return std::move(error);
      program.operations.push_back(std::move(inputWrite));
      emitTrace(trace, sequence, CIMTraceEventKind::ConfigureInput,
                group.groupId, work.workId, work.macroSlot, -1,
                input.inputCacheAddress);

      CIMDeviceOperation weightWrite;
      weightWrite.postDelayMs = policy.weightWriteDelayMs;
      llvm::SmallVector<uint32_t, 256> weightWords;
      weightWords.reserve(weight.words.size());
      for (int32_t word : weight.words)
        weightWords.push_back(llvm::bit_cast<uint32_t>(word));
      if (llvm::Error error = appendCIMWeightWordsPacket(
              weight.route, weightWords, weightWrite.flits))
        return std::move(error);
      program.operations.push_back(std::move(weightWrite));
      emitTrace(trace, sequence, CIMTraceEventKind::ConfigureWeight,
                group.groupId, work.workId, work.macroSlot);
    }
  }

  // The execution-plan contract completes every Core/Macro configuration
  // before triggering work. Keep all Once flits in one batch so the board
  // pacing delay begins only after every active Core has been started.
  CIMDeviceOperation startWork;
  startWork.postDelayMs = policy.workDelayMs;
  for (const CIMGroup &group : executable.getGroups()) {
    startWork.flits.push_back(encodeCIMWorkOnceFlit(group.works.front().route));
    emitTrace(trace, sequence, CIMTraceEventKind::StartWork, group.groupId);
  }
  program.operations.push_back(std::move(startWork));

  // The supplier BF16 testbench requires an explicit idle transition before a
  // response request. This path is candidate-only and disabled by default.
  if (hasBF16(executable)) {
    CIMDeviceOperation idle;
    for (const CIMGroup &group : executable.getGroups())
      idle.flits.push_back(encodeCIMWorkIdleFlit(group.works.front().route));
    program.operations.push_back(std::move(idle));
  }

  for (auto [outputIndex, readback] :
       llvm::enumerate(executable.getReadbacks())) {
    const CIMWork *work =
        findWork(executable, readback.groupId, readback.workId);
    if (!work)
      return runnerError("CIM readback references unknown work");
    CIMDeviceReadbackCommand command;
    command.outputIndex = outputIndex;
    command.dataType = readback.dataType;
    command.request.flits.push_back(encodeCIMControlFlit(
        readback.route, readback.macroSlot, readback.dataType));
    command.request.flits.push_back(encodeCIMOutputCacheReadFlit(
        readback.route, readback.outputCacheAddress));
    program.readbacks.push_back(std::move(command));
  }
  return program;
}

llvm::Error decodeResponse(const CIMTransaction &executable,
                           const CIMDeviceResponse &response,
                           CIMRunOutputs &outputs, CIMTraceSink *trace,
                           uint64_t &sequence) {
  if (response.readbacks.size() != executable.getReadbacks().size())
    return runnerError("CIM device returned an unexpected readback count");
  std::vector<bool> seen(outputs.values.size(), false);
  std::vector<std::array<int32_t, kCIMOutputElements>> decodedInt21(
      outputs.values.size());
  std::vector<std::array<uint16_t, kCIMOutputElements>> decodedBF16(
      outputs.values.size());

  for (const CIMDeviceReadback &result : response.readbacks) {
    if (result.outputIndex >= executable.getReadbacks().size() ||
        seen[result.outputIndex])
      return runnerError("CIM device returned an invalid readback index");
    seen[result.outputIndex] = true;
    const ReadbackBinding &binding =
        executable.getReadbacks()[result.outputIndex];
    const CIMWork *work = findWork(executable, binding.groupId, binding.workId);
    if (!work)
      return runnerError("CIM readback references unknown work");

    const uint64_t header = result.headerFlit;
    if (((header >> 60) & 0xf) != 5 || ((header >> 23) & 1) != 0 ||
        ((header >> 18) & 0x1f) != static_cast<uint64_t>(work->coreSlot) ||
        ((header >> 14) & 0xf) !=
            static_cast<uint64_t>(binding.outputCacheAddress) ||
        (header & 0x3fff) != 6)
      return runnerError(
          "CIM output Cache response head does not match binding");

    if (binding.dataType == CIMDataType::BF16) {
      const auto values =
          mlir::cim22::decodeBF16OutputCacheResponse(result.bodyFlits);
      std::copy(values.begin(), values.end(),
                decodedBF16[result.outputIndex].begin());
    } else {
      auto values = decodeCIMInt21OutputCacheResponse(result.bodyFlits);
      if (!values)
        return values.takeError();
      decodedInt21[result.outputIndex] = *values;
    }
    emitTrace(trace, sequence, CIMTraceEventKind::ReadOutput, binding.groupId,
              binding.workId, binding.macroSlot, -1,
              binding.outputCacheAddress);
  }
  for (size_t index = 0; index < outputs.values.size(); ++index) {
    const ReadbackBinding &binding = executable.getReadbacks()[index];
    if (binding.dataType == CIMDataType::BF16)
      std::copy(decodedBF16[index].begin(), decodedBF16[index].end(),
                outputs.values[index].bf16Values.begin());
    else
      std::copy(decodedInt21[index].begin(), decodedInt21[index].end(),
                outputs.values[index].values.begin());
  }
  return llvm::Error::success();
}
} // namespace

llvm::Error CIMBoardRunner::run(const execution::CIMTransaction &executable,
                                const execution::CIMRunInputs &inputs,
                                execution::CIMRunOutputs &outputs,
                                execution::CIMTraceSink *trace) {
  uint64_t sequence = 0;
  emitTrace(trace, sequence, execution::CIMTraceEventKind::BeginRun);
  if (llvm::Error error = validateViews(executable, inputs, outputs)) {
    emitTrace(trace, sequence, execution::CIMTraceEventKind::Error);
    return error;
  }
  if (hasBF16(executable) && !policy_.allowCandidateBF16) {
    emitTrace(trace, sequence, execution::CIMTraceEventKind::Error);
    return runnerError(
        "CIM board BF16 readback is disabled until response packing is verified "
        "(CTQ-037/HWQ-LAYOUT-003)");
  }
  auto program =
      materializeProgram(executable, inputs, policy_, trace, sequence);
  if (!program) {
    emitTrace(trace, sequence, execution::CIMTraceEventKind::Error);
    return program.takeError();
  }
  CIMDeviceResponse response;
  if (llvm::Error error = device_.execute(*program, response)) {
    emitTrace(trace, sequence, execution::CIMTraceEventKind::Error);
    return error;
  }
  if (llvm::Error error =
          decodeResponse(executable, response, outputs, trace, sequence)) {
    emitTrace(trace, sequence, execution::CIMTraceEventKind::Error);
    return error;
  }
  emitTrace(trace, sequence, execution::CIMTraceEventKind::Complete);
  return llvm::Error::success();
}

} // namespace cim22::runtime
